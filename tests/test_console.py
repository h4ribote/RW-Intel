"""What a person typing at the console is told.

The promise this file holds the console to is the one its own module states: a command is checked against the board as it currently stands and answered in a sentence, rather than failing, because someone naming a squad that was disbanded two periods ago is describing a board that has moved under them and a traceback says nothing about that. The second promise is that every command ends in one of the four operations the intervention mechanism offers — nothing here reaches the game by any other route — so what a command produces is a queued intervention and what a refusal produces is no intervention at all.

The server, the session and the policy are stubs, as they are in the intervention tests, because none of what is under test needs a game: a command is read, the board is consulted, and either a request joins the interface's queue or a sentence explains why none did.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rwintel.control.console import Console, HELP
from rwintel.control.intervention import Kind
from rwintel.control.policy.contracts import Doctrine, Posture, SquadRecord
from rwintel.wire import Commander


class _Strategy:
    def __init__(self) -> None:
        self.posture = Posture.EXPAND
        self.forced = None


class _Statistics:
    interventions = 0


class _Policy:
    """The chain, as far as the console reads it: the squad records, the strategic layer's posture and the count of interventions the status line reports."""

    def __init__(self, squads) -> None:
        self.squads = list(squads)
        self.strategy = _Strategy()
        self.statistics = _Statistics()
        self.catalogue = None


class _Session:
    def __init__(self, policy, instance: int = 0) -> None:
        self.instance = instance
        self.policy = policy
        #: No frame has arrived. Every command has to answer without one, since a console is opened before the first observation as often as after it.
        self.observation = None
        self.regions = []
        self.arm = "script"
        self.records = []
        self.episodes_wanted = 1


class _Server:
    def __init__(self, session) -> None:
        self.sessions = [session]
        self.stopped = False

    def session(self, instance: int):
        return next((s for s in self.sessions if s.instance == instance), None)

    def stop(self) -> None:
        self.stopped = True


def _squad(squad_id: int, members=(1, 2, 3), doctrine: Doctrine = Doctrine.VANGUARD,
           commander: int = 0) -> SquadRecord:
    return SquadRecord(id=squad_id, doctrine=doctrine, members=list(members), value=1000.0,
                       formed_value=1000.0, commander=commander)


def _console(*squads):
    """A console wired to one instance running the given squads, with its answers collected instead of printed."""
    said = []
    session = _Session(_Policy(squads))
    console = Console(_Server(session), write=said.append)
    interface = console.commander(session)
    return console, interface, said


def _queued(interface):
    """What is waiting to be drained onto the next action. Read here rather than drained because what a command produced is the question, and draining it would fold in the interface's own refusals as well."""
    return list(interface._pending)


def test_the_console_states_a_merge_as_an_intention_rather_than_as_a_list_of_units():
    console, interface, said = _console(_squad(0), _squad(1, members=[4, 5]))
    console.execute("merge 0 into 1")

    queued = _queued(interface)
    assert [request.kind for request in queued] == [Kind.REASSIGN]
    assert queued[0].squad == 0 and queued[0].into == 1
    # No units are named: which of them move is settled from the roster in hand when the request is drained, which is a period after this line was typed.
    assert queued[0].whole is True and list(queued[0].units) == []
    assert "every unit it holds then" in said[-1]


def test_a_merge_needs_the_word_that_says_which_squad_ceases_to_exist():
    console, interface, said = _console(_squad(0), _squad(1))
    console.execute("merge 0 1")

    assert _queued(interface) == []
    assert "merge 3 into 5" in said[-1]


def test_merging_a_squad_into_itself_is_refused_with_a_sentence():
    console, interface, said = _console(_squad(0))
    console.execute("merge 0 into 0")

    assert _queued(interface) == []
    assert "already itself" in said[-1]


def test_a_merge_that_names_a_squad_the_board_does_not_have_is_answered_with_the_squads_it_does():
    console, interface, said = _console(_squad(0), _squad(1))
    console.execute("merge 0 into 9")

    assert _queued(interface) == []
    assert "no squad 9" in said[-1] and "0, 1" in said[-1]


def test_a_squad_another_commander_holds_is_neither_merged_from_nor_into():
    """The observation says somebody outside holds it without saying who, which for the person typing is the distinction that matters: it is neither the chain's nor theirs, so its rosters are not theirs to change."""
    console, interface, said = _console(_squad(0, commander=int(Commander.OPERATIONS)), _squad(1))
    console.execute("merge 0 into 1")
    assert _queued(interface) == []
    assert "outside this console" in said[-1]

    console, interface, said = _console(_squad(0), _squad(1, commander=int(Commander.TACTICS)))
    console.execute("merge 0 into 1")
    assert _queued(interface) == []
    assert "squad 1 is held by a commander outside" in said[-1]


def test_a_squad_another_commander_holds_cannot_be_taken_either():
    """The guard belongs to taking as much as to the two verbs that reorganise, and not merely for symmetry. A queued take is remembered here as a holding asked for and not yet delivered, so one the interface is bound to refuse would have this console reporting a squad of somebody else's as its own for the rest of the episode — and would let the very merge just refused through, since a holding asked for counts as a holding."""
    console, interface, said = _console(_squad(0, commander=int(Commander.OPERATIONS)), _squad(1))
    console.execute("take 0")

    assert _queued(interface) == []
    assert "outside this console" in said[-1]
    assert 0 not in console.asked[0]

    console.execute("merge 0 into 1")
    assert _queued(interface) == []
    console.execute("status")
    assert "you hold nothing" in said[-1]


def test_merging_into_a_squad_of_your_own_reports_neither_a_doctrine_nor_a_settling_period():
    """Both of the sentences a merge into a squad of the chain's ends with are false of a squad this console raised. It sits in a slot the organisation layer has lent out and keeps no record of, so it is not reinforced at all and is never written an errand — and it has no doctrine either, the word shown in place of one being the word that says the squad is the person's own."""
    console, interface, said = _console(_squad(0, doctrine=Doctrine.VANGUARD))
    interface.own[3] = [7, 8]
    console.execute("merge 0 into 3")

    assert len(_queued(interface)) == 1
    assert "yours" not in said[-1]
    assert "will be reinforced as one" not in said[-1]
    assert "left alone for one operational period" not in said[-1]
    assert "one of your own" in said[-1]


def test_an_empty_squad_is_not_merged():
    console, interface, said = _console(_squad(0, members=[]), _squad(1))
    console.execute("merge 0 into 1")

    assert _queued(interface) == []
    assert "no members" in said[-1]


def test_merging_squads_of_different_doctrines_says_which_doctrine_survives():
    console, interface, said = _console(_squad(0, doctrine=Doctrine.RAID),
                                        _squad(1, members=[4], doctrine=Doctrine.VANGUARD))
    console.execute("merge 0 into 1")

    assert len(_queued(interface)) == 1
    assert "stays vanguard" in said[-1]


def test_a_merge_forgets_a_holding_asked_for_and_not_yet_in_hand():
    """A holding is remembered here between the moment it is asked for and the period that delivers it. Left behind for a squad that is about to cease to exist, it would report layers of whatever squad is raised into that number next."""
    console, interface, said = _console(_squad(0), _squad(1))
    console.execute("take 0")
    assert console.asked[0].get(0) == int(Commander.OPERATIONS)

    console.execute("merge 0 into 1")
    assert 0 not in console.asked[0]


def test_a_move_that_would_empty_a_squad_points_at_the_merge():
    """The refusal and the verb that answers it, pinned together so they cannot drift apart. A unit list typed against last period's listing is not a statement about a squad — the roster may have moved under the typist — so emptying a squad has to be said as an intention about the squad, which is a different sentence and so a different verb."""
    console, interface, said = _console(_squad(0, members=[1, 2, 3]), _squad(1))
    console.execute("move 0 1 2 3 to 1")

    assert _queued(interface) == []
    assert "merge 0 into M" in said[-1]


def test_merge_may_be_shortened_and_its_prefix_collisions_are_reported():
    console, interface, said = _console(_squad(0), _squad(1))
    console.execute("mer 0 into 1")
    assert len(_queued(interface)) == 1

    # 'fold' is what the organisation layer calls the same operation on its own squads, which makes it the natural second name for it.
    console.execute("fold 0 into 1")
    assert len(_queued(interface)) == 2

    console.execute("m 0 into 1")
    assert len(_queued(interface)) == 2
    assert "could be map, merge, move" in said[-1]


def test_every_command_the_help_names_is_one_the_console_answers_to():
    """The house rule that a document and the implementation may not differ, held mechanically on the one document a person reads from inside the program. Every indented line of the help shows a command; its first word is the verb, and each has to resolve and be answered."""
    console, _, said = _console(_squad(0))
    verbs = [line.split()[0] for line in HELP.splitlines()
             if line.startswith("  ") and line[2:3].strip()]

    assert "merge" in verbs and "move" in verbs
    for verb in verbs:
        assert verb in console._commands, verb
        before = len(said)
        console.execute(verb)
        assert len(said) > before, verb


if __name__ == "__main__":
    for name, function in sorted(globals().items()):
        if name.startswith("test_"):
            function()
            print("ok", name)
