import java.io.DataInputStream;
import java.io.EOFException;
import java.io.IOException;
import java.io.OutputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

/**
 * Framing and the fixed field layouts, the Java half of what `rwintel/wire` describes.
 * Any change here has to be made on both sides and has to raise the protocol version, because the two halves would otherwise keep talking while reading each other's bytes as something else.
 */
final class Wire {

    /** ASCII "RWIN", so a stream that has lost sync fails at the next header rather than silently. */
    static final int MAGIC = 0x4E495752;

    static final int PROTOCOL_VERSION = 2;

    static final int HEADER_SIZE = 16;

    static final int KIND_HELLO = 0x01;
    static final int KIND_EPISODE = 0x02;
    static final int KIND_OBSERVATION = 0x10;
    static final int KIND_ACTION = 0x20;
    static final int KIND_CONTROL = 0x30;

    static final int BLOCK_REGIONS = 1;
    static final int BLOCK_SQUADS = 2;
    static final int BLOCK_UNITS = 4;
    static final int BLOCK_EVENTS = 8;

    private Wire() {
    }

    static ByteBuffer buffer(int capacity) {
        return ByteBuffer.allocate(capacity).order(ByteOrder.LITTLE_ENDIAN);
    }

    /** Writes one frame. Callers hold the stream's monitor, because a frame must not be interleaved with another. */
    static void write(OutputStream out, int kind, int instance, byte[] body) throws IOException {
        ByteBuffer header = buffer(HEADER_SIZE);
        header.putInt(MAGIC);
        header.putShort((short) PROTOCOL_VERSION);
        header.putShort((short) kind);
        header.putShort((short) instance);
        header.putShort((short) 0);
        header.putInt(body.length);
        out.write(header.array());
        out.write(body);
        out.flush();
    }

    /** One received frame: the kind, and the body. */
    static final class Frame {
        final int kind;
        final byte[] body;

        Frame(int kind, byte[] body) {
            this.kind = kind;
            this.body = body;
        }
    }

    static Frame read(DataInputStream in) throws IOException {
        byte[] header = new byte[HEADER_SIZE];
        try {
            in.readFully(header);
        } catch (EOFException e) {
            return null;
        }
        ByteBuffer view = ByteBuffer.wrap(header).order(ByteOrder.LITTLE_ENDIAN);
        int magic = view.getInt();
        if (magic != MAGIC) throw new IOException("not a frame header: magic " + Integer.toHexString(magic));
        int version = view.getShort() & 0xFFFF;
        if (version != PROTOCOL_VERSION) {
            throw new IOException("protocol version " + version + ", expected " + PROTOCOL_VERSION);
        }
        int kind = view.getShort() & 0xFFFF;
        view.getShort();  // instance, which the game side already knows about itself
        view.getShort();  // flags, unused so far
        int length = view.getInt();
        byte[] body = new byte[length];
        if (length > 0) in.readFully(body);
        return new Frame(kind, body);
    }

    // ---- text bodies ---------------------------------------------------------------------

    /**
     * The smallest JSON writer that covers what the low rate frames carry.
     * A dependency for this would have to be shipped into the game's class path, and the alternative of hand rolling a format loses the readability that was the reason for choosing JSON for these frames at all.
     */
    static final class Json {
        private final StringBuilder text = new StringBuilder();
        private boolean needsComma = false;

        Json() {
            text.append('{');
        }

        private void separate() {
            if (needsComma) text.append(',');
            needsComma = true;
        }

        Json put(String key, String value) {
            separate();
            quote(key).append(':');
            if (value == null) text.append("null"); else quote(value);
            return this;
        }

        Json put(String key, long value) {
            separate();
            quote(key).append(':').append(value);
            return this;
        }

        Json put(String key, double value) {
            separate();
            quote(key).append(':');
            text.append(Double.isNaN(value) || Double.isInfinite(value) ? "0" : String.valueOf(value));
            return this;
        }

        Json put(String key, boolean value) {
            separate();
            quote(key).append(':').append(value);
            return this;
        }

        /** Inserts an already formatted array or object. */
        Json raw(String key, String json) {
            separate();
            quote(key).append(':').append(json);
            return this;
        }

        private StringBuilder quote(String value) {
            text.append('"');
            for (int i = 0; i < value.length(); i++) {
                char c = value.charAt(i);
                if (c == '"' || c == '\\') text.append('\\').append(c);
                else if (c == '\n') text.append("\\n");
                else if (c == '\r') text.append("\\r");
                else if (c == '\t') text.append("\\t");
                else if (c < 0x20) text.append(String.format("\\u%04x", (int) c));
                else text.append(c);
            }
            return text.append('"');
        }

        public String toString() {
            return text.toString() + "}";
        }

        byte[] toBytes() {
            try {
                return toString().getBytes("UTF-8");
            } catch (java.io.UnsupportedEncodingException e) {
                throw new IllegalStateException(e);
            }
        }
    }

    /**
     * Reads the few scalar fields the control frames carry.
     * Only flat objects of strings and numbers arrive here, so a full parser would be answering a question nobody asked.
     */
    static String field(String json, String key) {
        String needle = "\"" + key + "\"";
        int at = json.indexOf(needle);
        if (at < 0) return null;
        int colon = json.indexOf(':', at + needle.length());
        if (colon < 0) return null;
        int i = colon + 1;
        while (i < json.length() && Character.isWhitespace(json.charAt(i))) i++;
        if (i >= json.length()) return null;
        if (json.charAt(i) == '"') {
            StringBuilder out = new StringBuilder();
            for (i++; i < json.length(); i++) {
                char c = json.charAt(i);
                if (c == '\\' && i + 1 < json.length()) {
                    char next = json.charAt(++i);
                    if (next == 'n') out.append('\n');
                    else if (next == 'r') out.append('\r');
                    else if (next == 't') out.append('\t');
                    else out.append(next);
                } else if (c == '"') {
                    return out.toString();
                } else {
                    out.append(c);
                }
            }
            return out.toString();
        }
        int end = i;
        while (end < json.length() && "-+.eE0123456789truefalsnl".indexOf(json.charAt(end)) >= 0) end++;
        return json.substring(i, end);
    }

    static int intField(String json, String key, int fallback) {
        String value = field(json, key);
        if (value == null || value.isEmpty()) return fallback;
        try {
            return (int) Double.parseDouble(value);
        } catch (NumberFormatException e) {
            return fallback;
        }
    }

    static float floatField(String json, String key, float fallback) {
        String value = field(json, key);
        if (value == null || value.isEmpty()) return fallback;
        try {
            return Float.parseFloat(value);
        } catch (NumberFormatException e) {
            return fallback;
        }
    }

    static boolean boolField(String json, String key, boolean fallback) {
        String value = field(json, key);
        if (value == null || value.isEmpty()) return fallback;
        return "true".equalsIgnoreCase(value) || "1".equals(value);
    }
}
