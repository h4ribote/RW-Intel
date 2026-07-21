"""What the layers hand each other.

Every one of these is a structure with fixed meaning rather than a vector, and that is the whole design: a layer can be learnt, or replaced by a script, or taken over by a human, against frozen neighbours, because what crosses the boundary means the same thing to all three. A latent hand-off would tie the layers together so that changing one invalidates the other, and it would leave a human with nothing editable to edit.

The numbers here divide into two kinds. Some are structural — which fields exist, what a posture is, how many squads there may be — and changing one is a change of design. Others are opening values to be measured and adjusted, and are marked as such. The point of writing the whole command chain as a script first is that the second kind can be settled by measurement rather than by argument.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from ...wire.action import Commander, Stance, Status, Task


class Posture(enum.IntEnum):
    """What the strategic layer decides. Discrete rather than continuous because the point of this layer is that a human takes it over, and what a human wants to say is "we are arming now", not "0.62 against 0.38"."""

    EXPAND = 0
    TECH = 1
    ARM = 2
    DEFEND = 3
    DECIDE = 4


@dataclass(frozen=True)
class Allocation:
    """How the strategic layer wants the economy weighted. Shares of what is worth spending on, not a division of the credits themselves."""

    economy: float
    military: float
    tech: float


#: Read off the posture rather than emitted as numbers, so that the strategic action space stays five wide.
ALLOCATION: Dict[Posture, Allocation] = {
    Posture.EXPAND: Allocation(0.7, 0.2, 0.1),
    Posture.TECH: Allocation(0.4, 0.2, 0.4),
    Posture.ARM: Allocation(0.3, 0.6, 0.1),
    Posture.DEFEND: Allocation(0.4, 0.5, 0.1),
    Posture.DECIDE: Allocation(0.1, 0.8, 0.1),
}


class Role(enum.IntEnum):
    """What a unit is for. Decided from what the type can do, never from its name, so that a definition file that renames or replaces a built-in changes nothing here."""

    ARMOUR = 0
    ARTILLERY = 1
    ANTI_AIR = 2
    FAST = 3
    BUILDER = 4
    STRUCTURE = 5
    OTHER = 6


class Doctrine(enum.IntEnum):
    """What kind of squad this is. A squad's doctrine decides what it will accept as reinforcement and what it can be asked to do."""

    VANGUARD = 0
    GARRISON = 1
    RAID = 2
    ENGINEER = 3


@dataclass(frozen=True)
class DoctrineSpec:
    #: The smallest composition worth forming a squad from.
    minimum: Dict[Role, int]
    #: What a full squad of this kind looks like. The shortfall against this is what decides where a reinforcement goes.
    establishment: Dict[Role, int]
    #: Movement types this doctrine will take, so that a squad keeps to one kind of ground.
    movement: frozenset
    #: Tasks the operational layer may give a squad of this doctrine.
    tasks: tuple


DOCTRINES: Dict[Doctrine, DoctrineSpec] = {
    Doctrine.VANGUARD: DoctrineSpec(
        minimum={Role.ARMOUR: 4},
        establishment={Role.ARMOUR: 8, Role.ARTILLERY: 2},
        movement=frozenset({"LAND", "HOVER", "OVER_CLIFF"}),
        tasks=(Task.ATTACK, Task.ENCIRCLE),
    ),
    Doctrine.GARRISON: DoctrineSpec(
        minimum={Role.ARMOUR: 2},
        establishment={Role.ARMOUR: 4, Role.ANTI_AIR: 2},
        movement=frozenset({"LAND", "HOVER"}),
        tasks=(Task.DEFEND,),
    ),
    Doctrine.RAID: DoctrineSpec(
        minimum={Role.FAST: 3},
        establishment={Role.FAST: 6},
        movement=frozenset({"AIR", "HOVER"}),
        tasks=(Task.RAID, Task.WITHDRAW),
    ),
    Doctrine.ENGINEER: DoctrineSpec(
        minimum={Role.BUILDER: 1},
        establishment={Role.BUILDER: 3},
        movement=frozenset({"LAND", "HOVER"}),
        tasks=(Task.ESCORT,),
    ),
}

#: The most squads that may exist at once, which is what fixes the operational layer's action space and what the observation's squad block is sized for.
SQUAD_CAP = 8


@dataclass
class EconomyOrders:
    """Strategy to economy. What to weight spending towards, and what the army should end up looking like."""

    posture: Posture
    allocation: Allocation
    #: Credits per minute the economy may put into raising the technology level. Zero forbids it.
    tech_cap: float
    #: Shares of military spending by role, which the economy reads as what to build next.
    target_mix: Dict[Role, float]


@dataclass
class OperationsOrders:
    """Strategy to operations. Where matters, whether we are pressing, and what the whole of it may cost."""

    posture: Posture
    #: Region id to how much it is worth holding or taking, from 0 to 1.
    priorities: Dict[int, float]
    offensive: bool
    #: Credits of our own the strategic layer is prepared to lose across every mission at once.
    loss_allowance: float


@dataclass
class SquadRecord:
    """One squad, as the organisation layer keeps it and as the operational layer reads it. Its identity outlives any one mission, which is what lets a contract be about a squad rather than about a set of units."""

    id: int
    doctrine: Doctrine
    members: List[int] = field(default_factory=list)
    #: Filled in from the observation each period; absent until the game has reported the squad back.
    value: float = 0.0
    formed_value: float = 0.0
    x: float = 0.0
    y: float = 0.0
    spread: float = 0.0
    status: Status = Status.ACTIVE
    #: Which layers of this squad's command a human holds, as the bits the observation uses.
    commander: int = 0
    #: The contract it is under, if any.
    contract: Optional["TaskContract"] = None

    @property
    def machine(self) -> bool:
        """Whether nothing about this squad has been taken over. The organisation layer asks this, because a squad a human is steering at all is not one to reshuffle."""
        return self.commander == 0

    @property
    def ours_to_task(self) -> bool:
        """Whether the operational layer may still write this squad's contract. Taking only the tactical command leaves the contract ours to set, and the design calls the opposite split — a human writing contracts while the tactical layer fights them — the most useful arrangement of all."""
        return not self.commander & Commander.OPERATIONS

    @property
    def health(self) -> float:
        """Current worth against the most it has ever been worth, which is what the merge and disband rules are written in."""
        return 1.0 if self.formed_value <= 0 else self.value / self.formed_value


@dataclass
class TaskContract:
    """Operations to tactics. The one hand-off the design fixes field by field."""

    squad: int
    task: Task
    target_region: int
    stance: Stance
    #: Credits of our own this mission may spend. An absolute figure, not a share: a share has a denominator that moves while the mission runs, so a contract issued early would quietly grow.
    cost_budget: float
    #: The game time by which it should be done, absolute so that a period of delay does not change what it means.
    deadline_ms: int
    issued_at_ms: int


@dataclass
class MissionReport:
    """Tactics to operations. The only route by which anything above the fighting learns what is out there, which is why it exists even while the observation is omniscient."""

    squad: int
    status: Status
    #: What the squad has run into, by role and by worth.
    contact: Dict[Role, float] = field(default_factory=dict)
    losses: float = 0.0


@dataclass
class Shortfall:
    """Operations to organisation. Which squads cannot do what they are being asked to do."""

    squad: int
    #: Credits of strength the squad is short of its establishment.
    missing: float
    #: True when it is too worn to be given a mission at all, rather than merely under strength.
    worn_out: bool = False


@dataclass
class Replacement:
    """Organisation to economy. What has to be built for the squads to be whole."""

    role: Role
    count: int


@dataclass
class FrontReport:
    """Everything to strategy. The state of the front and of the economy, which is all the strategic layer decides on."""

    income: float
    credits: float
    military_value: float
    enemy_value: float
    #: Regions we hold a resource point in, and regions the enemy does.
    held: int
    enemy_held: int
    #: How many of our regions have changed hands away from us lately, which is what a posture change to defending is answering.
    lost_regions: int
    #: Regions with an enemy base still standing, as far as anything has seen.
    enemy_bases: int
