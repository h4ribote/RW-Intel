"""What a person actually types at.

The interface the design describes is not a second control path, and this is where that stops being a claim and becomes a program: every command here ends in one of the four operations the intervention mechanism offers, so what a person emits is a roster, a contract or a departure in exactly the fields the command chain emits them in. There is deliberately nothing here that reaches the game, and there could not be — this module holds an `Interface` and calls it, and the game side cannot tell a session a person amended from one the script intruder amended.

It is a line reader rather than a window, which is an ordering rather than a placeholder. What has to exist before a human's play is worth anything to the learning side is the operation set and the record of it; a window is a presentation of the same four operations and can be built against this once they have been settled by use. Reading stands on its own daemon thread because the work of the process is on the session threads and a blocked reader must not delay a period, and because a daemon thread lets the run end without the person having to press anything to release it.

Answers are sentences. Someone typing a squad that was disbanded two periods ago, or a region number off the end of the map, is describing a board that has moved under them, and a traceback says nothing about that; so every command checks what it is about against the board as it currently stands and reports what it found there instead of failing.

The board is read straight off the policy from this thread without a lock. That is safe in the only way it needs to be: each period replaces the squad list and the region table wholesale rather than mutating them in place, so a reader sees one period's list or the next one's and never half of either, and a list one period old is exactly what the person looking at the screen is reasoning about anyway.
"""

from __future__ import annotations

import logging
import sys
import threading
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from ..wire import Commander, Deviation, Stance, Status, Task
from .intervention import Interface, Recorder
from .policy.contracts import Posture

log = logging.getLogger(__name__)

#: How a person names the layers of a squad's command. Both the design's own words and the short forms someone types while a match is running, since the cost of an alias is a dictionary entry and the cost of refusing one is a command retyped at the worst moment.
LAYERS: Dict[str, Commander] = {
    "operations": Commander.OPERATIONS,
    "operational": Commander.OPERATIONS,
    "ops": Commander.OPERATIONS,
    "op": Commander.OPERATIONS,
    "tactics": Commander.TACTICS,
    "tactical": Commander.TACTICS,
    "tac": Commander.TACTICS,
    "both": Commander.OPERATIONS | Commander.TACTICS,
    "all": Commander.OPERATIONS | Commander.TACTICS,
}

#: What a contract written from here is funded with when no figure is given, as a share of what the squad is currently worth. A squad may spend itself, which is what a person means by "take that region" and the only default that does not require the board to be explained first.
DEFAULT_BUDGET_SHARE = 1.0

#: How long a contract written from here runs when no deadline is given, in game seconds. Long enough that a squad walking across the map is not judged before it has arrived.
DEFAULT_DEADLINE_S = 120

HELP = """\
Commands. Squads, units and regions are numbers; every name may be shortened to any unambiguous prefix.

  squads                            the squads of the current instance, and who holds each
  squad N                           one squad in full, with its members, which is where unit numbers come from
  regions                           the region table, ordered outward from home
  instance [N]                      which instance later commands apply to, or the list of them
  status                            what this console holds, and what the strategic layer is doing

  take N [ops|tactics|both]         take a squad's command; operations if no layer is named
  return N                          give it back, after which the organisation layer takes its units into the pool
  contract N TASK REGION [STANCE] [BUDGET] [SECONDS]
                                    write a squad's contract; stance aggressive, budget the squad's own worth, deadline {deadline}s
  move N UNIT... [to M]             move units out of squad N into squad M, or into a squad of your own if no M is given
  depart N HOW                      hold, withdraw, focus, spread or kite, for a squad whose tactical command you hold
  posture [NAME|auto]               pin the strategic layer to a posture, or hand it back its own judgement

  help                              this
  stop                              end the run

  tasks     {tasks}
  stances   {stances}
  postures  {postures}
""".format(
    deadline=DEFAULT_DEADLINE_S,
    tasks=" ".join(task.name.lower() for task in Task),
    stances=" ".join(stance.name.lower() for stance in Stance),
    postures=" ".join(posture.name.lower() for posture in Posture),
)


@dataclass
class Shown:
    """One squad as a person reads it, from whichever side of the process knows about it.

    Squads the chain formed are read from its own records, which carry the doctrine and the contract the operational layer wrote. Squads this console raised for itself exist in the interface and in the game and nowhere in the chain's records — the organisation layer deliberately does not look at a slot it has lent out — so they are read from the observation's squad block instead. Both are squads to the person holding them, so both are listed and both may be commanded, and the difference shows only as a doctrine there is nothing to report.
    """

    id: int
    doctrine: str
    members: List[int] = field(default_factory=list)
    value: float = 0.0
    health: float = 1.0
    status: str = "-"
    #: Who commands it, already phrased.
    held_by: str = "chain"
    #: Its errand, already phrased, or a dash where it has none.
    contract: str = "-"
    x: float = 0.0
    y: float = 0.0
    spread: float = 0.0
    #: True when this console raised the squad, which is what a command may not assume of a squad the chain formed and the reverse.
    raised: bool = False


class Console:
    """The reader, the per-instance interfaces it commands through, and the sentences it answers with.

    One `Interface` per instance, replaced at the start of every episode, because holding a squad is a fact about one match: a squad number means a different squad in the next episode and carrying a holding across would hand the person a squad they never took. The console keeps the current one under the instance number, so a command typed between episodes finds the instance and is told plainly that it is not running one, rather than acting on the match before.
    """

    def __init__(self, server, recorder: Optional[Recorder] = None,
                 write: Optional[Callable[[str], None]] = None) -> None:
        self.server = server
        self.recorder = recorder
        self._write = write or (lambda text: print(text, flush=True))
        #: The live interface per instance. Written from a session thread as an episode starts and read from this one; a whole value replaced under a key, which is why no lock guards it.
        self.interfaces: Dict[int, Interface] = {}
        #: Which instance an unqualified command is about. Set to the first instance to connect, since one instance is the ordinary case and asking a person to choose from a list of one is noise.
        self.instance: Optional[int] = None
        #: Layers asked for but not yet drained, by instance and squad. A request is queued and applied at the next period, so between typing `take 3 tactics` and that period the interface still reports the squad as the chain's; without this a person would be told they do not hold a squad they have just taken, and refused the very command the take was typed to enable. Ordering makes the refusal wrong as well as unhelpful: requests drain in the order they were made, so a departure queued behind its own take does apply.
        self.asked: Dict[int, Dict[int, int]] = {}
        self._thread: Optional[threading.Thread] = None
        self._commands = self._vocabulary()

    # ---- how it is plugged in ---------------------------------------------------------

    def commander(self, session) -> Interface:
        """The factory a server calls once per episode to build this console's outside commander for an instance."""
        interface = Interface(recorder=self.recorder, name="human", instance=session.instance)
        self.interfaces[session.instance] = interface
        self.asked[session.instance] = {}
        if self.instance is None:
            self.instance = session.instance
        return interface

    def start(self) -> None:
        self._thread = threading.Thread(target=self.run, name="console", daemon=True)
        self._thread.start()

    def run(self) -> None:
        self.say("Intervention console. Type help for the commands, stop to end the run.")
        # readline rather than iterating the file, because iteration reads ahead into a buffer and a line typed at a terminal would not be seen until the buffer had filled.
        for line in iter(sys.stdin.readline, ""):
            line = line.strip()
            if not line:
                continue
            try:
                self.execute(line)
            except Exception as error:
                # A command that fails is a fault in this file, not the end of the person's session, so the reader goes on to the next line.
                log.exception("console command failed: %s", line)
                self.say(f"that command failed: {error}")
        self.say("input has ended; the run carries on, and Ctrl-C or the agents finishing will end it.")

    def say(self, text: str) -> None:
        self._write(text)

    # ---- reading a line ---------------------------------------------------------------

    def execute(self, line: str) -> None:
        """One typed line. Public so that the vocabulary can be exercised without a terminal."""
        words = line.replace(",", " ").split()
        if not words:
            return
        name = words[0].lower()
        command = self._commands.get(name)
        if command is None:
            candidates = sorted(word for word in self._commands if word.startswith(name))
            # Distinct words that lead to the same command are not an ambiguity: 'return' and 'give' are one operation under two names, and refusing 'g' because both begin with it would be pedantry about a synonym.
            if len({self._commands[word] for word in candidates}) == 1 and candidates:
                command = self._commands[candidates[0]]
            elif candidates:
                self.say(f"{words[0]!r} could be {', '.join(candidates)}. Say more of it.")
                return
            else:
                self.say(f"I do not know {words[0]!r}. Type help for the commands.")
                return
        command(words[1:])

    def _vocabulary(self) -> Dict[str, Callable[[List[str]], None]]:
        return {
            "help": self.help, "?": self.help,
            "squads": self.squads, "list": self.squads,
            "squad": self.squad,
            "regions": self.regions, "map": self.regions,
            "instance": self.choose, "on": self.choose,
            "status": self.status,
            "take": self.take,
            "return": self.give_back, "give": self.give_back,
            "contract": self.contract, "order": self.contract,
            "move": self.move, "detach": self.move,
            "depart": self.depart, "deviate": self.depart,
            "posture": self.posture,
            "stop": self.stop, "quit": self.stop, "exit": self.stop,
        }

    # ---- what it can be asked ---------------------------------------------------------

    def help(self, words: List[str]) -> None:
        self.say(HELP)

    def choose(self, words: List[str]) -> None:
        live = self._instances()
        if not words:
            if not live:
                self.say("no instance has connected yet.")
                return
            current = "none chosen" if self.instance is None else f"currently {self.instance}"
            self.say(f"connected: {', '.join(str(i) for i in live)} ({current}).")
            return
        instance = _number(words[0])
        if instance is None:
            self.say(f"{words[0]!r} is not an instance number.")
            return
        if instance not in live:
            self.say(f"no instance {instance} is connected; connected: "
                     f"{', '.join(str(i) for i in live) or 'none'}.")
            return
        self.instance = instance
        self.say(f"commands now apply to instance {instance}.")

    def status(self, words: List[str]) -> None:
        session = self._session()
        if session is None:
            return
        interface = self.interfaces.get(session.instance)
        policy = session.policy
        lines = [f"instance {session.instance}, arm {session.arm or '-'}, "
                 f"{len(session.records)} episode(s) finished of {session.episodes_wanted}"]
        if policy is None:
            lines.append("no episode is running, so there is nothing to command yet")
        else:
            lines.append(f"strategic posture {policy.strategy.posture.name.lower()} "
                         f"({'pinned by you' if policy.strategy.forced is not None else 'the layer decides'})")
            lines.append(f"{policy.statistics.interventions} intervention(s) applied this episode, "
                         f"counting everyone outside the chain")
        asked = self.asked.get(session.instance, {})
        squads = sorted(set(interface.held) | set(interface.own) | set(asked)) if interface else []
        if squads:
            for squad in squads:
                raised = ", raised by you" if squad in interface.own else ""
                pending = ("" if squad in interface.held or squad in interface.own
                           else ", asked for and not yet in hand")
                lines.append(f"squad {squad}: you hold "
                             f"{_layers_name(self._holding(session, interface, squad))}{raised}{pending}")
        else:
            lines.append("you hold nothing")
        if self.recorder is not None:
            lines.append(f"interventions are being recorded to {self.recorder.path}")
        self.say("\n".join(lines))

    def squads(self, words: List[str]) -> None:
        running = self._running()
        if running is None:
            return
        session, policy, interface = running
        observation = session.observation
        now = observation.game_time_ms if observation is not None else 0
        head = (f"instance {session.instance} at {now // 1000}s"
                + (f", credits {observation.credits:.0f}" if observation is not None else "")
                + f", posture {policy.strategy.posture.name.lower()}"
                + ("  [pinned]" if policy.strategy.forced is not None else ""))
        rows = ["{:>3} {:<9} {:>4} {:>8} {:>6} {:<8} {:<13} {}".format(
            "id", "doctrine", "size", "worth", "health", "status", "held by", "contract")]
        for shown in self._shown(session, policy, interface):
            rows.append("{:>3} {:<9} {:>4} {:>8.0f} {:>6.2f} {:<8} {:<13} {}".format(
                shown.id, shown.doctrine, len(shown.members), shown.value, shown.health,
                shown.status, shown.held_by, shown.contract))
        if len(rows) == 1:
            rows.append("(no squad has been formed yet)")
        self.say(head + "\n" + "\n".join(rows))

    def squad(self, words: List[str]) -> None:
        running = self._running()
        if running is None:
            return
        session, policy, interface = running
        shown = self._one(session, policy, interface, words[0] if words else "")
        if shown is None:
            return
        lines = [f"squad {shown.id}, {shown.doctrine}, {len(shown.members)} unit(s) worth "
                 f"{shown.value:.0f} at {shown.health:.0%} of its best, {shown.status}, "
                 f"held by {shown.held_by}",
                 f"contract: {shown.contract}",
                 f"at ({shown.x:.0f}, {shown.y:.0f}), spread {shown.spread:.0f}"]
        observation = session.observation
        by_id = {unit.id: unit for unit in observation.unit_states} if observation is not None else {}
        for member in shown.members:
            unit = by_id.get(member)
            if unit is None:
                lines.append(f"  unit {member}, not in the last observation")
                continue
            kind = policy.catalogue.kind(unit.type_index)
            health = unit.health / unit.max_health if unit.max_health else 0.0
            lines.append(f"  unit {member} {kind.lookup if kind is not None else '?'} "
                         f"at {health:.0%} health")
        if not shown.members:
            lines.append("  (no members)")
        self.say("\n".join(lines))

    def regions(self, words: List[str]) -> None:
        session = self._session()
        if session is None:
            return
        if not session.regions:
            self.say(f"instance {session.instance} has no region table yet; the map has not been reported.")
            return
        ordered = session.ordered_regions()
        observation = session.observation
        states = {state.id: state for state in observation.regions} if observation is not None else {}
        placing = ("ordered outward from home, which is the order the layers see them in"
                   if session.home is not None
                   else "in the map's own order, because nothing has been built yet and home is not known")
        rows = ["{:>6} {:>4} {:>9} {:>5} {:>6} {:>8}".format(
            "region", "rank", "resources", "ours", "theirs", "distance")]
        for rank, region in enumerate(ordered):
            state = states.get(region.id)
            rows.append("{:>6} {:>4} {:>9} {:>5} {:>6} {:>8}".format(
                region.id, rank, region.resources,
                state.held_by_us if state is not None else "-",
                state.held_by_enemy if state is not None else "-",
                f"{state.distance_from_home:.0f}" if state is not None else "-"))
        self.say(f"regions of instance {session.instance}, {placing}.\n"
                 f"Type the number in the 'region' column. That is the map's own numbering, which is what "
                 f"travels on the wire and what a contract's target means; 'rank' is only how far out a "
                 f"region is, 0 being home, and is typed nowhere.\n" + "\n".join(rows))

    def take(self, words: List[str]) -> None:
        running = self._running()
        if running is None:
            return
        session, policy, interface = running
        shown = self._one(session, policy, interface, words[0] if words else "")
        if shown is None:
            return
        layers = Commander.OPERATIONS
        if len(words) > 1:
            named = LAYERS.get(words[1].lower())
            if named is None:
                self.say(f"{words[1]!r} is not a layer. Say operations, tactics or both.")
                return
            layers = named
        if shown.raised:
            self.say(f"squad {shown.id} is one of your own and already answers to you alone.")
            return
        interface.take(shown.id, int(layers))
        asked = self.asked.setdefault(session.instance, {})
        # Taking replaces what is held rather than adding to it, since that is what the interface does with the layers it is given.
        asked[shown.id] = int(layers)
        self.say(f"taking {_layers_name(int(layers))} of squad {shown.id} at the next period; "
                 f"the chain stops writing about it from that moment.")

    def give_back(self, words: List[str]) -> None:
        running = self._running()
        if running is None:
            return
        session, _, interface = running
        squad = _number(words[0] if words else "")
        if squad is None:
            self.say("say which squad to give back, as 'return 3'.")
            return
        if not self._holding(session, interface, squad):
            self.say(f"you are not holding squad {squad}, so there is nothing to give back.")
            return
        interface.give_back(squad)
        self.asked.get(session.instance, {}).pop(squad, None)
        self.say(f"giving squad {squad} back. The organisation layer will empty it and place its units "
                 f"through the ordinary reinforcement rule, since where you left them is not knowable here.")

    def contract(self, words: List[str]) -> None:
        running = self._running()
        if running is None:
            return
        session, policy, interface = running
        if len(words) < 3:
            self.say("a contract needs at least a squad, a task and a region, as 'contract 3 attack 5'.")
            return
        shown = self._one(session, policy, interface, words[0])
        if shown is None:
            return
        task = _pick(Task, words[1])
        if task is None:
            self.say(f"{words[1]!r} is not a task. One of: {', '.join(t.name.lower() for t in Task)}.")
            return
        region = self._region(session, words[2])
        if region is None:
            return
        stance = Stance.AGGRESSIVE
        if len(words) > 3:
            named = _pick(Stance, words[3])
            if named is None:
                self.say(f"{words[3]!r} is not a stance. One of: "
                         f"{', '.join(s.name.lower() for s in Stance)}.")
                return
            stance = named
        budget = _decimal(words[4]) if len(words) > 4 else shown.value * DEFAULT_BUDGET_SHARE
        if budget is None:
            self.say(f"{words[4]!r} is not an amount of credits.")
            return
        seconds = _decimal(words[5]) if len(words) > 5 else float(DEFAULT_DEADLINE_S)
        if seconds is None:
            self.say(f"{words[5]!r} is not a number of seconds.")
            return
        now = session.observation.game_time_ms if session.observation is not None else 0
        interface.write(shown.id, task=task, target_region=region, stance=stance,
                        cost_budget=budget, deadline_ms=now + int(seconds * 1000))
        holding = self._holding(session, interface, shown.id) & int(Commander.OPERATIONS)
        self.say(f"squad {shown.id}: {task.name.lower()} region {region}, stance {stance.name.lower()}, "
                 f"spending up to {budget:.0f} credits, due in {seconds:.0f}s. "
                 + ("Nothing will overwrite it, since you hold its operational command."
                    if holding or shown.raised else
                    "The squad stays under the chain, which will fight the new errand until the "
                    "operational layer has a reason to write another."))

    def move(self, words: List[str]) -> None:
        running = self._running()
        if running is None:
            return
        session, policy, interface = running
        into = -1
        if len(words) > 2 and words[-2].lower() in ("to", "into"):
            destination = self._one(session, policy, interface, words[-1])
            if destination is None:
                return
            into = destination.id
            words = words[:-2]
        if len(words) < 2:
            self.say("say which squad and which units, as 'move 3 101 102' or 'move 3 101 102 to 5'.")
            return
        shown = self._one(session, policy, interface, words[0])
        if shown is None:
            return
        units = [_number(word) for word in words[1:]]
        if any(unit is None for unit in units):
            self.say("every unit has to be a number; 'squad 3' lists them.")
            return
        strangers = [unit for unit in units if unit not in shown.members]
        if strangers:
            self.say(f"unit(s) {', '.join(str(u) for u in strangers)} are not in squad {shown.id}. "
                     f"Its members are {', '.join(str(m) for m in shown.members) or 'none'}.")
            return
        if len(units) >= len(shown.members):
            self.say(f"that is every unit in squad {shown.id}; take the squad whole with "
                     f"'take {shown.id} both' rather than emptying it.")
            return
        interface.reassign(shown.id, units, into=into)
        if into >= 0:
            self.say(f"moving {len(units)} unit(s) from squad {shown.id} into squad {into} "
                     f"at the next period.")
        else:
            self.say(f"detaching {len(units)} unit(s) from squad {shown.id} into a squad of your own. "
                     f"Which number it is given is settled when the request is drained, so ask for "
                     f"'squads' afterwards; if all eight slots are in use the move will not happen.")

    def depart(self, words: List[str]) -> None:
        running = self._running()
        if running is None:
            return
        session, _, interface = running
        if len(words) < 2:
            self.say("say which squad and how, as 'depart 3 withdraw'.")
            return
        squad = _number(words[0])
        if squad is None:
            self.say(f"{words[0]!r} is not a squad number.")
            return
        deviation = _pick(Deviation, words[1])
        if deviation is None:
            self.say(f"{words[1]!r} is not a departure. One of: "
                     f"{', '.join(d.name.lower() for d in Deviation)}.")
            return
        if not self._holding(session, interface, squad) & int(Commander.TACTICS):
            self.say(f"you do not hold the tactical command of squad {squad}, so how it moves is the "
                     f"tactical layer's to choose. Say 'take {squad} tactics' first.")
            return
        interface.depart(squad, deviation)
        self.say(f"squad {squad} will {deviation.name.lower()}.")

    def posture(self, words: List[str]) -> None:
        running = self._running()
        if running is None:
            return
        _, policy, _ = running
        if not words:
            state = ("pinned there by you" if policy.strategy.forced is not None
                     else "chosen by the strategic layer")
            self.say(f"posture {policy.strategy.posture.name.lower()}, {state}.")
            return
        word = words[0].lower()
        if word in ("auto", "release", "none", "off"):
            policy.strategy.forced = None
            self.say("the strategic layer decides its own posture again, from its next period.")
            return
        posture = _pick(Posture, word)
        if posture is None:
            self.say(f"{words[0]!r} is not a posture. One of: "
                     f"{', '.join(p.name.lower() for p in Posture)}, or 'auto' to hand it back.")
            return
        policy.strategy.forced = posture
        self.say(f"the strategic layer is pinned to {posture.name.lower()} until you say 'posture auto'.")

    def stop(self, words: List[str]) -> None:
        self.say("ending the run.")
        self.server.stop()

    # ---- finding what a command is about ----------------------------------------------

    def _instances(self) -> List[int]:
        return sorted(session.instance for session in self.server.sessions if session.instance >= 0)

    def _session(self):
        live = self._instances()
        if not live:
            self.say("no instance has connected yet, so there is nothing to command.")
            return None
        instance = self.instance
        if instance not in live:
            if len(live) > 1:
                self.say(f"instances {', '.join(str(i) for i in live)} are connected; "
                         f"say 'instance N' to choose which these commands are about.")
                return None
            instance = live[0]
            self.instance = instance
        session = self.server.session(instance)
        if session is None:
            self.say(f"instance {instance} is no longer connected.")
        return session

    def _running(self):
        """The session, its policy and this console's interface to it, or nothing with a sentence saying why not."""
        session = self._session()
        if session is None:
            return None
        if session.policy is None:
            self.say(f"instance {session.instance} is not running an episode yet, "
                     f"so there is nothing to take command of.")
            return None
        interface = self.interfaces.get(session.instance)
        if interface is None:
            self.say(f"instance {session.instance} was not given a console interface, "
                     f"so nothing typed here would reach it.")
            return None
        return session, session.policy, interface

    def _holding(self, session, interface: Interface, squad: int) -> int:
        """Which layers of a squad this console has, counting what it has asked for and the next period has not yet delivered."""
        if squad in interface.own:
            return int(Commander.OPERATIONS | Commander.TACTICS)
        return interface.held.get(squad, 0) | self.asked.get(session.instance, {}).get(squad, 0)

    def _shown(self, session, policy, interface: Interface) -> List[Shown]:
        """Every squad on this side, the chain's and this console's own, in one list ordered by number."""
        now = session.observation.game_time_ms if session.observation is not None else 0
        states = ({state.id: state for state in session.observation.squads}
                  if session.observation is not None else {})
        squads = [_from_record(record, self._holding(session, interface, record.id), now)
                  for record in policy.squads]
        known = {shown.id for shown in squads}
        for squad, members in sorted(interface.own.items()):
            if squad not in known:
                squads.append(_from_state(squad, members, states.get(squad), now))
        return sorted(squads, key=lambda shown: shown.id)

    def _one(self, session, policy, interface: Interface, word: str) -> Optional[Shown]:
        squad = _number(word)
        if squad is None:
            self.say(f"{word!r} is not a squad number.")
            return None
        squads = self._shown(session, policy, interface)
        for shown in squads:
            if shown.id == squad:
                return shown
        known = ", ".join(str(shown.id) for shown in squads)
        self.say(f"there is no squad {squad}. The squads are {known or 'none yet'}.")
        return None

    def _region(self, session, word: str) -> Optional[int]:
        region = _number(word)
        if region is None:
            self.say(f"{word!r} is not a region number.")
            return None
        known = {entry.id for entry in session.regions}
        if not known:
            self.say("the region table has not arrived yet, so a target cannot be checked.")
            return None
        if region not in known:
            self.say(f"region {region} is not on this map, whose regions run {min(known)} to "
                     f"{max(known)}. Ask for 'regions' to see them.")
            return None
        return region


def _from_record(record, holding: int, now: int) -> Shown:
    contract = record.contract
    return Shown(
        id=record.id, doctrine=record.doctrine.name.lower(), members=list(record.members),
        value=record.value, health=record.health, status=record.status.name.lower(),
        held_by=_holder(holding, record.commander),
        contract=("-" if contract is None else
                  _contract_text(contract.task, contract.target_region, contract.stance,
                                 contract.cost_budget, contract.deadline_ms, now)),
        x=record.x, y=record.y, spread=record.spread,
    )


def _from_state(squad: int, members: List[int], state, now: int) -> Shown:
    """A squad this console raised, described from the game's own squad block. The chain keeps no record of one, so the observation is the only thing that knows what it is currently worth or where it stands."""
    if state is None:
        return Shown(id=squad, doctrine="yours", members=list(members), held_by="you: both", raised=True)
    return Shown(
        id=squad, doctrine="yours", members=list(members), value=state.value,
        health=1.0 if state.formed_value <= 0 else state.value / state.formed_value,
        status=Status(state.status).name.lower(), held_by="you: both",
        contract=_contract_text(Task(state.task_type), state.target_region, Stance(state.stance),
                                state.cost_budget, state.deadline_ms, now),
        x=state.x, y=state.y, spread=state.spread, raised=True,
    )


def _number(word: str) -> Optional[int]:
    try:
        return int(word)
    except (TypeError, ValueError):
        return None


def _decimal(word: str) -> Optional[float]:
    try:
        return float(word)
    except (TypeError, ValueError):
        return None


def _pick(kind, word: str):
    """A member of an enumeration from what a person typed: its own name, any unambiguous prefix of it, or its number. Prefixes are taken because these names are long and someone is typing them while a match runs."""
    word = (word or "").strip().lower()
    if not word:
        return None
    number = _number(word)
    if number is not None:
        try:
            return kind(number)
        except ValueError:
            return None
    by_name = {member.name.lower(): member for member in kind}
    if word in by_name:
        return by_name[word]
    matches = [member for name, member in by_name.items() if name.startswith(word)]
    return matches[0] if len(matches) == 1 else None


def _layers_name(layers: int) -> str:
    if layers & int(Commander.OPERATIONS) and layers & int(Commander.TACTICS):
        return "both layers"
    if layers & int(Commander.OPERATIONS):
        return "the operational command"
    if layers & int(Commander.TACTICS):
        return "the tactical command"
    return "nothing"


def _holder(holding: int, commander: int) -> str:
    """Who commands a squad, in the width a table can carry. The observation says which layers are in someone else's hands but never whose, so what this console holds is what names the holder, and the bare bits are the fallback — which is how a squad the script intruder has taken shows up, and correctly so, since to a person it is simply not theirs and not the chain's."""
    if holding:
        return f"you: {_layers_short(holding)}"
    if commander:
        return f"outside: {_layers_short(commander)}"
    return "chain"


def _layers_short(layers: int) -> str:
    if layers & int(Commander.OPERATIONS) and layers & int(Commander.TACTICS):
        return "both"
    return "ops" if layers & int(Commander.OPERATIONS) else "tac"


def _contract_text(task: Task, region: int, stance: Stance, budget: float,
                   deadline_ms: int, now: int) -> str:
    if not deadline_ms:
        return "-"
    return (f"{task.name.lower()} region {region} {stance.name.lower()} "
            f"budget {budget:.0f} due {(deadline_ms - now) / 1000.0:+.0f}s")
