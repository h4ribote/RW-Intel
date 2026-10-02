import java.io.File;
import java.util.Arrays;
import java.util.Locale;
import java.util.Set;
import java.util.TreeMap;
import java.util.TreeSet;

/**
 * Starts, watches and ends skirmish episodes on instruction from the control process.
 *
 * The battleroom route is used rather than the quick start: beginning a match with no network session makes the engine redraw the seed and flatten the income multiplier, so the settings asked for would not be the settings played.
 *
 * Each episode rebuilds the server. The engine's own return to the battleroom cannot be used here, because its countdown is driven from the network tick and that tick stops being taken part way through a single player match, leaving the timer armed forever. Rebuilding costs a map load but always lands in a known state, and it also clears the player slots that would otherwise accumulate.
 */
final class MatchDriver {

    /** The team the engine gives to players who watch rather than play. */
    private static final int SPECTATOR = -3;

    private static final String SKIRMISH_DIRECTORY = "assets/maps/skirmish";

    /** How long a networked host holds its room open for the other process when not told otherwise. Generous, because what it is waiting through is that process loading the game. */
    static final int PEER_WAIT_SECONDS = 90;

    /** Wall clock a hosted match goes on with nobody connected to it before it is called abandoned. Long enough to ride out a connection being replaced, short beside a match. */
    private static final int PEER_GONE_MS = 5000;

    /** Game time that has to pass before the match is judged at all, which is long enough for every player's starting units to have been placed. */
    private static final int START_GRACE_MS = 3000;

    static final class Settings {
        String map = "";
        int opponents = 1;
        int difficulty = 1;
        /** Leaves this many AI players in the match and moves everyone else, including the local player, to the spectators. Zero leaves the room alone. */
        int contestants = 0;
        /** Sends the built-in AI players' orders on operational observations. */
        boolean aiOrders = false;
        /** Which AI player, by slot order (the order contestants are kept in), the episode is observed from, with the answers kept as books; -1 for none. */
        int watch = -1;
        int credits = 0;
        int startingUnits = 1;
        float income = 1.0f;
        int fog = 2;
        int seed = 12345;
        int maxSeconds = 0;
        /** Game milliseconds between the standings sent while an ordinary episode runs, or 0 to send none. They are what a finished match is read back from at a moment before it was decided. */
        int standingMs = 0;
        /**
         * Holds the episode open even when only one side has anything on the board.
         *
         * An episode used to construct engagements in starts with no units at all, so every player is wiped from the first frame and the ordinary end test would finish the match before a single unit had been spawned into it. What ends such an episode is the clock or the control process saying so, and nothing else.
         */
        boolean arena = false;
        /**
         * Opens a real multiplayer session rather than the single player server, so that a second process can join this match before it starts.
         *
         * Nothing else about the episode changes: the map, the room, the AI opponents and the seed are all settled the same way. That is deliberate, because the point of running a match over a session two processes share is to find out whether the ordinary episode stays in step, not to run a different episode.
         */
        boolean networked = false;
        /** The port a networked host binds. Two hosts on one machine need different ones, since the engine reads this from the settings as it binds rather than taking it as an argument. */
        int networkPort = 5123;
        /** Seconds a networked host holds its room open for somebody to join, or 0 to hold it until somebody does. A person joining from their own client may take any time; another process should not. */
        int peerWaitSeconds = PEER_WAIT_SECONDS;
        /** host[:port] of a match to join instead of hosting one. A joining process settles nothing about the match: the map, the settings, the seed and the moment of the start all arrive from the host. */
        String joinAddress = "";
        /** The name this process answers to in a session. It is what the host's per client desync report names each client by, so it is worth making it say which process this is. */
        String name = "";
    }

    private final Engine engine;
    private Settings settings = new Settings();
    private String resolvedMap = "";
    private int episode = 0;

    MatchDriver(Engine engine) {
        this.engine = engine;
    }

    Settings settings() {
        return settings;
    }

    String map() {
        return resolvedMap;
    }

    int episode() {
        return episode;
    }

    /** Resolves the requested map substring to a path the engine accepts. Names contain spaces, which an agent argument cannot. */
    private String resolveMap(String wanted) {
        File directory = new File(SKIRMISH_DIRECTORY);
        String[] names = directory.list();
        if (names == null) throw new IllegalStateException("no map directory at " + directory.getAbsolutePath());
        Arrays.sort(names);
        String needle = wanted.toLowerCase(Locale.ENGLISH);
        for (String name : names) {
            if (!name.endsWith(".tmx")) continue;
            if (name.toLowerCase(Locale.ENGLISH).contains(needle)) return "maps/skirmish/" + name;
        }
        throw new IllegalStateException("no built-in map matching '" + wanted + "'");
    }

    /**
     * Brings up a server on the requested map and starts the match, unless it has to wait for somebody first.
     *
     * Returns whether the match actually began. A networked host does not begin until the other process is registered as a player in its room, which cannot be waited for from here: registration is the far end of a handshake the engine works through on its own threads and its own loop, and holding the game thread still to wait for it is holding still the thing that has to run for it to finish. So the room is left standing and {@link #beginWhenReady} is polled instead, one poll per step, until there is somebody to start against.
     */
    boolean start(Object game, Settings requested) throws Exception {
        settings = requested;
        stopReplay(game);
        replay = null;
        resolvedMap = resolveMap(requested.map);
        waitFailure = null;
        aloneSinceMs = 0;
        peerLeft = false;

        Object net = engine.net(game);
        engine.invoke(net, "b", String.class, "rw-intel setup");
        engine.resetPlayers();
        engine.invoke(engine.getField(game, "bS"), "g");
        engine.invoke(game, "L");

        synchronized (game) {
            engine.setField(game, "dm", null);
            engine.setField(game, "dl", resolvedMap);
        }

        Object normal = engine.staticField(engine.loadModeClass, "b");
        engine.loadModeClass.getClass();
        java.lang.reflect.Method load = game.getClass().getMethod("a", boolean.class, engine.loadModeClass);
        load.setAccessible(true);
        load.invoke(game, Boolean.TRUE, normal);

        engine.setField(net, "y", playerName());
        engine.setField(net, "o", Boolean.TRUE);
        openServer(game, net);

        Object config = engine.getField(net, "ay");
        engine.setField(config, "a", engine.staticField(engine.mapKindClass, "a"));
        engine.setField(net, "az", resolvedMap);
        engine.setField(config, "b", resolvedMap.substring(resolvedMap.lastIndexOf('/') + 1));
        engine.setField(config, "c", Integer.valueOf(settings.credits));
        engine.setField(config, "d", Integer.valueOf(settings.fog));
        engine.setField(config, "e", Boolean.FALSE);
        engine.setField(config, "f", Integer.valueOf(settings.difficulty));
        engine.setField(config, "g", Integer.valueOf(settings.startingUnits));
        engine.setField(config, "h", Float.valueOf(settings.income));
        engine.setField(config, "i", Boolean.FALSE);
        engine.setField(config, "l", Boolean.FALSE);

        // A networked host takes its opponent from the other process rather than manufacturing one, so no AI is added: on a map for two there is one other slot and the peer needs it.
        if (!settings.networked) {
            for (int i = 0; i < settings.opponents; i++) engine.invoke(net, "ap");
        }
        engine.invoke(net, "f");
        engine.invoke(net, "P");
        engine.invoke(net, "L");

        // Taken out after the room has finished populating itself, because it fills every free slot with an AI whatever was asked for.
        if (settings.contestants > 0) chooseContestants();
        if (settings.arena) chooseSparringPartner();

        if (settings.networked) {
            openedAtMs = System.currentTimeMillis();
            return beginWhenReady(game);
        }
        begin(game, net);
        return true;
    }

    /**
     * Starts a networked match once somebody else is in the room, and says whether it has started.
     *
     * Called once a step while a host is waiting, so that the engine's own loop keeps running and the handshake that turns a connection into a player can finish. A connection is not enough to start against: it exists from the moment the socket is accepted, several exchanges before the far end has a name, a slot and a place on the map.
     *
     * A room nobody joined within the wait is closed, which releases the match port, and {@link #waitFailure} then says why no match began.
     */
    boolean beginWhenReady(Object game) throws Exception {
        if (!settings.networked || openedAtMs == 0) return false;
        if (peers() == 0) {
            if (settings.peerWaitSeconds > 0 && System.currentTimeMillis() - openedAtMs > settings.peerWaitSeconds * 1000L) {
                openedAtMs = 0;
                engine.invoke(engine.net(game), "b", String.class, "rw-intel nobody joined");
                waitFailure = "no other player joined within " + settings.peerWaitSeconds + "s";
            }
            return false;
        }
        openedAtMs = 0;
        begin(game, engine.net(game));
        return true;
    }

    /** Why a networked host gave up waiting and closed its room, or null while it has not. */
    String waitFailure() {
        return waitFailure;
    }

    private String waitFailure = null;

    /** Players in the room other than this process's own, which is what a host is waiting for and what says the handshake finished rather than merely started. */
    private int peers() throws Exception {
        Object local = engine.local(engine.engine());
        int found = 0;
        int slots = engine.slotCount();
        for (int i = 0; i < slots; i++) {
            Object player = engine.playerAt(i);
            if (player != null && player != local) found++;
        }
        return found;
    }

    private void begin(Object game, Object net) throws Exception {
        // The seed is written last, because both of the calls that bring a server up redraw it: returning to the battleroom does, and so does opening a session for others to join.
        Object config = engine.getField(net, "ay");
        engine.setField(config, "q", Integer.valueOf(settings.seed));
        engine.invoke(net, "ae");
        episode++;
    }

    /** When a networked host opened its room, or nought when it is not waiting for anybody. */
    private long openedAtMs = 0;

    /**
     * Brings up the server the match will run on, which is the one place a networked host differs from a single player one.
     *
     * A single player server is the whole session inside this process. Nothing can join it, so the engine never has a second world to compare its own against and the lockstep machinery it carries is never exercised at all. A networked host binds a port and admits other processes, and from that point on it checksums the world for each of them and records what each one answers.
     */
    private void openServer(Object game, Object net) throws Exception {
        if (!settings.networked) {
            if (!Boolean.TRUE.equals(engine.invoke(net, "S"))) {
                throw new IllegalStateException("single player server did not start");
            }
            return;
        }
        if (settings.networkPort < 1024 || settings.networkPort > 65535) {
            throw new IllegalStateException("network port out of range: " + settings.networkPort);
        }
        if (!engine.hostNetworked(game, settings.networkPort)) {
            throw new IllegalStateException("could not host on port " + settings.networkPort
                    + ", which usually means another process is already holding it");
        }
    }

    /**
     * Leaves an arena episode with one opponent that does not think, so that both sides of a constructed engagement are driven from the control process and by nothing else.
     *
     * A board on which engagements are constructed has to be a board on which nothing else is happening, and the room does not offer one. The starting-unit setting has no effect through this start sequence: every value produced a command centre and a builder for each player with a starting position, so an empty board cannot simply be asked for. What can be arranged is that the only other player left in the match is one the map has no starting position for. The room fills nine slots whatever was asked for, and on a map for two, everything past the second has nowhere to appear; the last of those is the sparring partner. Everyone else, the player with the second base included, goes to the spectators so that no second match is played in the background.
     *
     * Choosing the slot is not enough on its own, and this is the part measurement had to teach. A player with no base is still a computer player, and it thinks: it forms attack groups out of whatever it owns, gives them orders of its own, and finds itself an economy. Measured over ten game minutes with eight instances, the opposing side finished with an income of a hundred and sixteen credits a second and forty-seven thousand credits of units, against a side that was only ever given what the arena spawned for it, and won twice as often from the same starting strength. So every computer player in the room is halted through the flag its own update reads before doing anything. It stops deciding without a unit, a health or a credit being touched, which leaves the units the arena spawns answerable to the control process alone. With the halt the opposing side ends an episode with no income at all.
     */
    private void chooseSparringPartner() throws Exception {
        int slots = engine.slotCount();
        Object local = engine.local(engine.engine());
        int partner = -1;
        for (int i = slots - 1; i >= 0; i--) {
            Object player = engine.playerAt(i);
            if (player == null || player == local) continue;
            partner = i;
            break;
        }
        int halted = 0;
        for (int i = 0; i < slots; i++) {
            Object player = engine.playerAt(i);
            if (player == null || player == local) continue;
            // Every computer player in the room is stopped from thinking, the sparring partner included. A partner with nowhere to appear has no base to build from, but it is still a computer player, and one that thinks will form its own attack groups out of the units the arena spawns for it and give them orders of its own. The opposing side of a constructed engagement has to be driven by the layer under study and by nothing else, or what is measured is that layer against itself plus a second opinion.
            if (engine.haltAi(player)) halted++;
            if (i != partner) engine.setTeam(player, SPECTATOR);
        }
        if (partner >= 0 && local != null && engine.team(engine.playerAt(partner)) == engine.team(local)) {
            engine.setTeam(engine.playerAt(partner), engine.team(local) + 1);
        }
        sparringSlot = partner;
        RwAgent.log("arena: sparring slot " + partner + " of " + slots + " slot(s), " + halted + " computer player(s) halted");
    }


    /** The slot an arena episode's opposing side is spawned for, or -1 outside an arena episode. The control process is told, because it is what decides which player each constructed unit belongs to. */
    int sparringSlot() {
        return settings.arena ? sparringSlot : -1;
    }

    private int sparringSlot = -1;

    /**
     * Plays a recorded match back, and says whether the replay loaded.
     *
     * The replay is looked for in the replays folder of the working directory. Loading it replaces the world with the save it starts from; the menu is then closed as the game's own replay load closes it, because an open menu document holds the simulation at frame 0. The world then steps through the recorded commands on its own, and nothing this process sends reaches it.
     *
     * The playback ends when the replay player stops, or at `untilMs` of game time when that is given. A recording can run on past the end of its match, and the defeat the recording side saw is not reproduced on playback, so the end of the match is the caller's to say.
     */
    boolean playReplay(Object game, String name, int untilMs, int steps) throws Exception {
        settings = new Settings();
        stopReplay(game);
        replay = null;
        waitFailure = null;
        aloneSinceMs = 0;
        peerLeft = false;
        if (!engine.loadReplay(game, name)) return false;
        engine.closeMenu();
        if (steps > 1) engine.setReplaySteps(game, steps);
        replay = name;
        replayUntilMs = untilMs;
        Object path = engine.getField(game, "dl");
        resolvedMap = path == null ? "" : String.valueOf(path);
        return true;
    }

    /** Counts a loaded playback as the episode now under way, once it is certain to be observed. */
    void countReplay() {
        episode++;
    }

    /** The replay being played back, or null during a match. */
    String replay() {
        return replay;
    }

    /**
     * Why the playback now running is over, or null while it goes on.
     *
     * Running out of recorded commands is not the end: the engine marks the replay ended when its file runs out, and the world goes on stepping without further commands, which is exactly what the recorded match did if nobody issued any.
     */
    String replayFinished(Object game) throws Exception {
        if (!engine.replayActive(game)) return "stopped";
        if (replayUntilMs > 0 && engine.gameTime(game) >= replayUntilMs) return "until";
        return null;
    }

    /** Whether the playback has read to the end of its file. */
    boolean replayExhausted(Object game) throws Exception {
        return replay != null && engine.replayEnded(game);
    }

    /** The slot of the one playing side not on `team`, or -1 when there is not exactly one. A match against a person is recorded with the policy's team, and the person is the other side. */
    int onlyOpponentOf(int team) throws Exception {
        int found = -1;
        int slots = engine.slotCount();
        for (int i = 0; i < slots; i++) {
            Object player = engine.playerAt(i);
            if (player == null || engine.team(player) < 0 || engine.team(player) == team) continue;
            if (found >= 0) return -1;
            found = engine.slot(player);
        }
        return found;
    }

    /** Ends the playback, so that its file is released and the world stops stepping. */
    void stopReplay(Object game) throws Exception {
        if (replay != null && engine.replayActive(game)) engine.stopReplay(game);
    }

    private String replay = null;
    private int replayUntilMs = 0;

    /**
     * Closes the recording the engine keeps of a match, and says which file it was written to, or null when nothing was being recorded.
     *
     * The engine goes on recording a hosted match after it has been decided, for as long as the process runs; closed here, the file holds the match and nothing after it.
     */
    String closeRecording(Object game) throws Exception {
        if (replay != null || !engine.replayRecording(game)) return null;
        String file = engine.replayFile(game);
        engine.stopReplay(game);
        return file;
    }

    /**
     * Joins a match another process is hosting.
     *
     * Nothing is loaded and nothing is configured here, because none of it is this process's to decide. The host chooses the map, the settings and the seed and sends all of them over as it starts, so what follows a successful join is a wait, and {@link #running} is what ends it.
     *
     * A host compares a checksum of the core unit definitions before it admits anyone and refuses a client whose units differ, so both processes have to be running the same install with the same mods, which for instances that share one master copy means simply that neither of them was given any.
     */
    void join(Object game, Settings requested) throws Exception {
        settings = requested;
        resolvedMap = "";
        engine.setNetworkName(game, playerName());
        String failure = engine.join(game, requested.joinAddress);
        if (failure != null) {
            throw new IllegalStateException("could not join " + requested.joinAddress + ": " + failure);
        }
    }

    /**
     * Takes note that a match this process joined has begun, reading back what the host decided about it.
     *
     * The episode is counted here rather than at the join, because a connection that is never followed by a start is not an episode and counting it would leave the two sides disagreeing about how many have been run.
     */
    void joined(Object game) throws Exception {
        Object path = engine.getField(engine.net(game), "az");
        resolvedMap = path == null ? "" : String.valueOf(path);
        episode++;
    }

    /** Whether the engine has a match in progress. For a process that joined one, this going true is the only sign that the host has started it. */
    boolean running(Object game) throws Exception {
        return engine.getBoolean(engine.net(game), "aW");
    }

    /** The seed the match is actually running under, which for a process that joined one is the host's rather than anything asked for here. */
    int seed(Object game) throws Exception {
        Object config = engine.getField(engine.net(game), "ay");
        Object value = config == null ? null : engine.getField(config, "q");
        return value instanceof Integer ? ((Integer) value).intValue() : settings.seed;
    }

    /**
     * What the engine knows about whether the processes in this session are still simulating the same match.
     *
     * The engine settles this itself rather than leaving it to be guessed at from the log. The host checksums the world every few hundred frames, each client answers with its own, and the host keeps every client's verdict on the connection it arrived over. It matters to a run because a session that has drifted apart does not stop: it goes on playing, only no longer the same match on each side, so any number taken out of an episode after that moment describes nothing.
     */
    String synchronisation(Object game) throws Exception {
        StringBuilder peers = new StringBuilder("[");
        for (Object connection : engine.connections(game)) {
            Wire.Json peer = new Wire.Json();
            peer.put("name", engine.peerName(connection));
            peer.put("desynced", engine.peerDesynced(connection));
            peer.put("broken", engine.peerBroken(connection));
            peer.put("matched", engine.peerMatches(connection));
            peer.put("desyncs", engine.peerDesyncs(connection));
            if (peers.length() > 1) peers.append(',');
            peers.append(peer.toString());
        }
        peers.append(']');

        Wire.Json report = new Wire.Json();
        report.put("networked", engine.networked(game));
        report.put("host", engine.isHost(game));
        report.put("frame", engine.checksumFrame(game));
        report.put("interval", engine.checksumInterval(game));
        report.put("checksum", engine.checksum(game));
        report.put("matched", engine.checksumMatches(game));
        report.raw("peers", peers.toString());
        return report.toString();
    }

    /** The name this process plays under. The engine's own default is used when nothing was asked for, so that a single player episode looks exactly as it did before there was anything to name. */
    private String playerName() {
        return settings.name.isEmpty() ? "You" : settings.name;
    }

    /** The slot of the `index`-th AI player in slot order, which with contestants kept is the `index`-th contestant; -1 when there is none. */
    int aiSlot(int index) throws Exception {
        int seen = 0;
        for (int i = 0; i < engine.slotCount(); i++) {
            Object player = engine.playerAt(i);
            if (player == null || !engine.isAi(player)) continue;
            if (seen++ == index) return engine.slot(player);
        }
        return -1;
    }

    private void chooseContestants() throws Exception {
        int slots = engine.slotCount();
        int kept = 0;
        for (int i = 0; i < slots; i++) {
            Object player = engine.playerAt(i);
            if (player == null) continue;
            if (kept < settings.contestants && engine.isAi(player)) {
                engine.setTeam(player, kept++);
            } else {
                engine.setTeam(player, SPECTATOR);
            }
        }
        if (kept < settings.contestants) {
            throw new IllegalStateException("only " + kept + " AI players available");
        }
    }

    /** True once the match cannot usefully continue: one side left, the engine has called it, or the time limit is up. */
    boolean finished(Object game) throws Exception {
        if (replay != null) return replayFinished(game) != null;
        if (!running(game)) return true;
        if (settings.maxSeconds > 0 && engine.gameTime(game) / 1000 >= settings.maxSeconds) return true;
        if (abandoned(game)) {
            peerLeft = true;
            return true;
        }
        // An arena episode is a board to build situations on rather than a match to win, and every test below asks who is winning. Its engagements are begun and ended by the control process, and it runs until the clock or an abort stops it.
        if (settings.arena) return false;
        // Nothing is decided in the first moments of a match. A player whose starting units have not been placed yet reads as wiped out, so the count of surviving sides is one until everybody is on the board; without this a match ends before it begins as soon as a second process is in it and its player is registered a frame later than this one's.
        if (engine.gameTime(game) < START_GRACE_MS) return false;
        // The engine's victory and defeat flags speak for the local player, which means nothing once that player is watching.
        if (settings.contestants == 0 && (engine.victory(game) || engine.defeat(game))) return true;
        return aliveTeams().size() <= 1;
    }

    /**
     * Whether a match this process hosts has had nobody connected to it for {@link #PEER_GONE_MS}.
     *
     * A player who leaves keeps their team on the board, so the ordinary end test would run the match on to its cutoff against an opponent nobody is commanding.
     */
    private boolean abandoned(Object game) throws Exception {
        if (!settings.networked || !engine.isHost(game)) return false;
        if (!engine.connections(game).isEmpty()) {
            aloneSinceMs = 0;
            return false;
        }
        long now = System.currentTimeMillis();
        if (aloneSinceMs == 0) aloneSinceMs = now;
        return now - aloneSinceMs >= PEER_GONE_MS;
    }

    /** Whether everybody who joined the hosted match just finished had left it by the end, whichever test ended it. */
    boolean peerLeft(Object game) throws Exception {
        return peerLeft || (settings.networked && engine.isHost(game) && engine.connections(game).isEmpty());
    }

    /** When a hosted match was first seen with nobody connected, or nought while somebody is. */
    private long aloneSinceMs = 0;

    private boolean peerLeft = false;

    Set<Integer> aliveTeams() throws Exception {
        int slots = engine.slotCount();
        Set<Integer> teams = new TreeSet<Integer>();
        for (int i = 0; i < slots; i++) {
            Object player = engine.playerAt(i);
            if (player == null) continue;
            int team = engine.team(player);
            if (team == SPECTATOR) continue;
            if (engine.defeated(player) || engine.wiped(player) || engine.surrendered(player)) continue;
            teams.add(Integer.valueOf(team));
        }
        return teams;
    }

    /**
     * Where each playing side stands, which is what a match with no winner has to be scored from.
     *
     * Four quantities per team, because the design scores an unfinished match on all of them: what is standing and what it is worth, what is coming in, what has been traded, and the credits held unspent. None of them can be worked out from outside the process for the enemy, so all of them go over here.
     * Every non-spectator team is listed, including the teams of slots nobody plays from; those read nought throughout, which is how the control side tells them from a side in the match. A player that never owned a unit, earned, killed or lost anything adds no credits to its team, so the starting credits of a slot the map has no start for do not count for the side it happens to share a team with.
     */
    String standing(Object game) throws Exception {
        TreeMap<Integer, long[]> byTeam = new TreeMap<Integer, long[]>();
        java.util.Set<Object> owning = java.util.Collections.newSetFromMap(new java.util.IdentityHashMap<Object, Boolean>());
        Object[] units = engine.unitArray();
        int count = engine.unitCount();
        for (int i = 0; i < count && i < units.length; i++) {
            Object unit = units[i];
            if (unit == null || engine.dead(unit)) continue;
            Object owner = engine.owner(unit);
            if (owner == null) continue;
            int team = engine.team(owner);
            // Negative teams are the spectators and the neutral owner that holds the scenery, neither of which is a side in the match.
            if (team < 0) continue;
            if (engine.built(unit) < 1f) continue;
            owning.add(owner);
            long[] tally = tallyFor(byTeam, team);
            tally[0]++;
            tally[1] += engine.price(unit);
        }

        int slots = engine.slotCount();
        for (int i = 0; i < slots; i++) {
            Object player = engine.playerAt(i);
            if (player == null) continue;
            int team = engine.team(player);
            if (team < 0) continue;
            long[] tally = tallyFor(byTeam, team);
            long income = (long) engine.income(player);
            Object record = engine.record(game, player);
            long killed = engine.recordInt(record, "c") + engine.recordInt(record, "d");
            long lost = engine.recordInt(record, "f") + engine.recordInt(record, "g");
            tally[2] += income;
            tally[3] += killed;
            tally[4] += lost;
            // A player the room made for a slot the map has no start for stands nowhere and spends nothing, but still holds the starting credits; it can share a team with a player in the match, and its credits are then not that side's.
            if (owning.contains(player) || income != 0 || killed != 0 || lost != 0) tally[5] += (long) engine.credits(player);
        }

        StringBuilder out = new StringBuilder("[");
        for (java.util.Map.Entry<Integer, long[]> entry : byTeam.entrySet()) {
            long[] tally = entry.getValue();
            if (out.length() > 1) out.append(',');
            out.append("{\"team\":").append(entry.getKey())
                    .append(",\"units\":").append(tally[0])
                    .append(",\"value\":").append(tally[1])
                    .append(",\"income\":").append(tally[2])
                    .append(",\"killed\":").append(tally[3])
                    .append(",\"lost\":").append(tally[4])
                    .append(",\"credits\":").append(tally[5]).append('}');
        }
        return out.append(']').toString();
    }

    private static long[] tallyFor(TreeMap<Integer, long[]> byTeam, int team) {
        long[] tally = byTeam.get(Integer.valueOf(team));
        if (tally == null) byTeam.put(Integer.valueOf(team), tally = new long[6]);
        return tally;
    }

    String players() throws Exception {
        int slots = engine.slotCount();
        StringBuilder out = new StringBuilder("[");
        for (int i = 0; i < slots; i++) {
            Object player = engine.playerAt(i);
            if (player == null) continue;
            if (out.length() > 1) out.append(',');
            Wire.Json entry = new Wire.Json();
            entry.put("slot", engine.slot(player));
            entry.put("team", engine.team(player));
            entry.put("ai", engine.isAi(player));
            entry.put("level", engine.aiLevel(player));
            entry.put("name", engine.name(player));
            out.append(entry.toString());
        }
        return out.append(']').toString();
    }
}
