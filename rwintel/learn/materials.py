"""What a state was encoded from, kept so that it can be encoded again.

A recorded state is only as useful as the encoding that wrote it, and the encoding is the part of this project most likely to change. Each decision therefore keeps, beside the state, the things the state was made from: for the tactical layer the squad, its members, what threatens it and the ground it was sent to; for the operational layer the whole board, the orders, the squads, and where the squad decided about could get to and by which transport; for the economy the board it costs its investments on and the offers themselves. Run through the encoding in force (`rebuild`), they give the state that encoding would have written, so changing a feature costs a pass over the data rather than the data.

What is taken at the moment of the decision is only references and shallow copies, since that moment is on the thread a game is waiting on; the rows are made from them on the writing thread (`tables`). Things that are the same for every decision of a run -the type table and the combat table the predictions come from- are written once per run (`context`), because the combat table reads a report that grows from run to run and a state rebuilt against a later one would not be the state that was played.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields, replace
from typing import Dict, List, Optional, Sequence, Tuple

from ..control.policy.catalogue import Catalogue
from ..control.policy.combat import CombatTable
from ..control.policy.contracts import Doctrine, Domain, OperationsOrders, Posture, Role, SquadRecord, TaskContract
from ..control.policy.encoding import (
    Access,
    EconomicBoard,
    Investment,
    INVESTMENT_SLOTS,
    Offer,
    TransportView,
    economic_state,
    forces,
    operational_state,
    tactical_state,
)
from ..control.policy.view import Sighting, WorldView, build
from ..wire import PASSAGE_CLASSES, Observation, RegionState, Stance, Status, Task, UnitState

#: The columns of a unit row, a region row and a squad row, in order.
UNIT_COLUMNS = tuple(f.name for f in fields(UnitState))
REGION_COLUMNS = tuple(f.name for f in fields(RegionState))
SQUAD_COLUMNS = ("id", "doctrine", "value", "formed_value", "x", "y", "spread", "losses", "status", "commander",
                 "settling", "contract", "task", "target_region", "stance", "cost_budget", "deadline_ms",
                 "issued_at_ms", "domain", "aboard")

_UNIT_FLOATS = {"x", "y", "health", "max_health"}
_REGION_FLOATS = {"x", "y", "our_value", "enemy_value", "distance_from_home"}

#: The columns of the operational board row: the clock, the observation's own totals, the home region, the squad decided about, the orders (posture, whether pressing, the allowance, whether there were orders at all), and whether the squad's access was known.
BOARD_COLUMNS = ("now", "credits", "income", "units", "unit_cap", "under_construction", "home", "squad",
                 "posture", "offensive", "allowance", "orders", "access")

#: The columns of a transport slot row; the movement is written as its index in `PASSAGE_CLASSES`.
TRANSPORT_COLUMNS = tuple(f.name for f in fields(TransportView))
_TRANSPORT_FLOATS = {"x", "y"}
_TRANSPORT_BOOLS = {"valid", "busy", "mine", "carries"}

#: The economy's board, less the target mix, which is kept as rows of its own.
ECONOMIC_BOARD_COLUMNS = tuple(f.name for f in fields(EconomicBoard) if f.name != "target_mix")
OFFER_COLUMNS = ("slot",) + tuple(f.name for f in fields(Offer) if f.name != "payload")
_OFFER_ENUMS = {"kind": Investment, "role": Role}
_OFFER_BOOLS = {f.name for f in fields(Offer) if f.type in ("bool", bool)}
_OFFER_INTS = {f.name for f in fields(Offer) if f.type in ("int", int)}
_BOARD_BOOLS = {f.name for f in fields(EconomicBoard) if f.type in ("bool", bool)}
_BOARD_INTS = {f.name for f in fields(EconomicBoard) if f.type in ("int", int)}

#: The tables each layer's materials are written as, each a list of rows of the named columns.
TABLES: Dict[str, Dict[str, Tuple[str, ...]]] = {
    "tactics": {"squad": SQUAD_COLUMNS, "members": UNIT_COLUMNS, "threats": UNIT_COLUMNS, "target": REGION_COLUMNS,
                "fight": ("losses", "killed", "now")},
    "operations": {"board": BOARD_COLUMNS, "units": UNIT_COLUMNS, "regions": REGION_COLUMNS,
                   "squads": SQUAD_COLUMNS, "members": ("squad_row", "unit"), "priorities": ("region", "priority"),
                   "expansion": ("region",), "spawns": ("region",), "access": ("region", "walk", "coastal"),
                   "lifts": ("region", "slot"), "transports": TRANSPORT_COLUMNS},
    "economy": {"board": ECONOMIC_BOARD_COLUMNS, "mix": ("role", "share"), "offers": OFFER_COLUMNS},
}


# ---- taking them, on the deciding thread ---------------------------------------------------

def _squad_copy(squad: SquadRecord) -> SquadRecord:
    """A copy that later periods cannot change under the record. Contracts are replaced rather than edited, so the one held is kept by reference."""
    return replace(squad, members=list(squad.members))


@dataclass
class TacticalMaterials:
    squad: SquadRecord
    members: List[UnitState]
    threats: List[UnitState]
    target: Optional[RegionState]
    losses: float
    killed: float
    now: int


def tactical(squad: SquadRecord, members: Sequence[Sighting], threats: Sequence[Sighting], losses: float,
             killed: float, view: WorldView, now: int) -> TacticalMaterials:
    target = view.region(squad.contract.target_region) if squad.contract is not None else None
    return TacticalMaterials(squad=_squad_copy(squad), members=[s.unit for s in members],
                             threats=[s.unit for s in threats], target=target, losses=losses, killed=killed, now=now)


@dataclass
class OperationalMaterials:
    observation: Observation
    regions: List[RegionState]
    home: int
    orders: Optional[OperationsOrders]
    squads: List[SquadRecord]
    spawns: Tuple[int, ...]
    squad: int
    now: int
    access: Optional[Access] = None


def operational(view: WorldView, orders: Optional[OperationsOrders], squads: Sequence[SquadRecord],
                spawns: Sequence[int], squad: SquadRecord, now: int,
                access: Optional[Access] = None) -> OperationalMaterials:
    return OperationalMaterials(observation=view.observation, regions=list(view.regions),
                                home=view.home.id if view.home is not None else -1, orders=orders,
                                squads=[_squad_copy(s) for s in squads], spawns=tuple(spawns), squad=squad.id, now=now,
                                access=access)


@dataclass
class EconomicMaterials:
    board: EconomicBoard
    slots: List[Optional[Offer]]


def economic(board: EconomicBoard, slots: Sequence[Optional[Offer]]) -> EconomicMaterials:
    return EconomicMaterials(board=replace(board, target_mix=dict(board.target_mix)),
                             slots=[replace(offer, payload=None) if offer is not None else None for offer in slots])


# ---- writing them as rows, on the writing thread -------------------------------------------

def _unit_row(unit: UnitState) -> List[float]:
    return [float(getattr(unit, name)) for name in UNIT_COLUMNS]


def _region_row(region: RegionState) -> List[float]:
    return [float(getattr(region, name)) for name in REGION_COLUMNS]


def _squad_row(squad: SquadRecord) -> List[float]:
    contract = squad.contract
    row = [squad.id, int(squad.doctrine), squad.value, squad.formed_value, squad.x, squad.y, squad.spread,
           squad.losses, int(squad.status), squad.commander, 1.0 if squad.settling else 0.0]
    if contract is None:
        row.extend([0.0] * 7)
    else:
        row.extend([1.0, int(contract.task), contract.target_region, int(contract.stance), contract.cost_budget,
                    contract.deadline_ms, contract.issued_at_ms])
    row.extend([int(squad.domain), squad.aboard])
    return [float(value) for value in row]


def _transport_row(transport: TransportView) -> List[float]:
    values = {name: getattr(transport, name) for name in TRANSPORT_COLUMNS}
    values["movement"] = PASSAGE_CLASSES.index(transport.movement) if transport.movement in PASSAGE_CLASSES else 0
    return [float(values[name]) for name in TRANSPORT_COLUMNS]


def _access_tables(access: Optional[Access], regions: Sequence[RegionState]) -> Dict[str, List[List[float]]]:
    if access is None:
        return {"access": [], "lifts": [], "transports": []}
    return {"access": [[float(r.id), 1.0 if r.id in access.walk else 0.0, 1.0 if r.id in access.coastal else 0.0]
                       for r in regions],
            "lifts": [[float(region), float(slot)] for region, slots in sorted(access.lift.items()) for slot in slots],
            "transports": [_transport_row(t) for t in access.transports]}


def tables(materials) -> Dict[str, List[List[float]]]:
    """One decision's materials as rows, table by table (`TABLES`)."""
    if isinstance(materials, TacticalMaterials):
        return {"squad": [_squad_row(materials.squad)], "members": [_unit_row(u) for u in materials.members],
                "threats": [_unit_row(u) for u in materials.threats],
                "target": [_region_row(materials.target)] if materials.target is not None else [],
                "fight": [[float(materials.losses), float(materials.killed), float(materials.now)]]}
    if isinstance(materials, OperationalMaterials):
        observation, orders = materials.observation, materials.orders
        board = [materials.now, observation.credits, observation.income, observation.units, observation.unit_cap,
                 observation.under_construction, materials.home, materials.squad,
                 int(orders.posture) if orders is not None else 0, 1.0 if orders is not None and orders.offensive else 0.0,
                 orders.loss_allowance if orders is not None else 0.0, 1.0 if orders is not None else 0.0,
                 1.0 if materials.access is not None else 0.0]
        members = [[float(row), float(unit)] for row, squad in enumerate(materials.squads) for unit in squad.members]
        return {"board": [[float(value) for value in board]],
                "units": [_unit_row(u) for u in observation.unit_states],
                "regions": [_region_row(r) for r in materials.regions],
                "squads": [_squad_row(s) for s in materials.squads], "members": members,
                "priorities": [[float(region), float(value)] for region, value in sorted((orders.priorities if orders else {}).items())],
                "expansion": [[float(region)] for region in (orders.expansion if orders else [])],
                "spawns": [[float(region)] for region in materials.spawns],
                **_access_tables(materials.access, materials.regions)}
    if isinstance(materials, EconomicMaterials):
        board = materials.board
        return {"board": [[float(int(v) if isinstance(v, (bool, Posture)) else v)
                           for v in (getattr(board, name) for name in ECONOMIC_BOARD_COLUMNS)]],
                "mix": [[float(int(role)), float(share)] for role, share in sorted(board.target_mix.items())],
                "offers": [[float(slot)] + [float(int(v) if isinstance(v, (bool, Investment, Role)) else v)
                                            for v in (getattr(offer, name) for name in OFFER_COLUMNS[1:])]
                           for slot, offer in enumerate(materials.slots) if offer is not None]}
    raise TypeError(f"no materials of type {type(materials).__name__}")


# ---- encoding them again -------------------------------------------------------------------

@dataclass
class Context:
    """What every decision of a run was encoded against: the type table and the combat table built from it."""

    catalogue: Catalogue
    combat: Optional[CombatTable]

    @classmethod
    def of(cls, types: Sequence[dict], combat: Optional[dict]) -> "Context":
        from ..control.session import UnitType

        catalogue = Catalogue.of_types([UnitType(**entry) for entry in types])
        return cls(catalogue=catalogue, combat=CombatTable.from_snapshot(combat, catalogue) if combat else None)


def _named(row: Sequence[float], columns: Sequence[str]) -> Dict[str, float]:
    return dict(zip(columns, row))


def _unit(row: Sequence[float]) -> UnitState:
    values = _named(row, UNIT_COLUMNS)
    return UnitState(**{name: (float(v) if name in _UNIT_FLOATS else int(v)) for name, v in values.items()})


def _region(row: Sequence[float]) -> RegionState:
    values = _named(row, REGION_COLUMNS)
    return RegionState(**{name: (float(v) if name in _REGION_FLOATS else int(v)) for name, v in values.items()})


def _squad(row: Sequence[float], members: Sequence[int] = ()) -> SquadRecord:
    v = _named(row, SQUAD_COLUMNS)
    contract = None
    if v["contract"]:
        contract = TaskContract(squad=int(v["id"]), task=Task(int(v["task"])), target_region=int(v["target_region"]),
                                stance=Stance(int(v["stance"])), cost_budget=float(v["cost_budget"]),
                                deadline_ms=int(v["deadline_ms"]), issued_at_ms=int(v["issued_at_ms"]))
    return SquadRecord(id=int(v["id"]), doctrine=Doctrine(int(v["doctrine"])), members=list(members),
                       value=float(v["value"]), formed_value=float(v["formed_value"]), x=float(v["x"]),
                       y=float(v["y"]), spread=float(v["spread"]), losses=float(v["losses"]),
                       status=Status(int(v["status"])), commander=int(v["commander"]), settling=bool(v["settling"]),
                       contract=contract, domain=Domain(int(v["domain"])), aboard=int(v["aboard"]))


def _transport(row: Sequence[float]) -> TransportView:
    v = _named(row, TRANSPORT_COLUMNS)
    values = {name: (float(x) if name in _TRANSPORT_FLOATS else bool(x) if name in _TRANSPORT_BOOLS else int(x))
              for name, x in v.items()}
    values["movement"] = PASSAGE_CLASSES[int(v["movement"])]
    return TransportView(**values)


def _access(board: Dict[str, float], rows: Dict[str, Sequence[Sequence[float]]]) -> Optional[Access]:
    if not board["access"]:
        return None
    lift: Dict[int, List[int]] = {}
    for region, slot in rows["lifts"]:
        lift.setdefault(int(region), []).append(int(slot))
    return Access(walk={int(r) for r, walk, _ in rows["access"] if walk},
                  lift=lift, coastal={int(r) for r, _, coastal in rows["access"] if coastal},
                  transports=[_transport(row) for row in rows["transports"]])


def _sighting(unit: UnitState, catalogue: Catalogue) -> Sighting:
    return Sighting(unit=unit, kind=catalogue.kind(unit.type_index), role=catalogue.role(unit.type_index))


def rebuild(layer: str, rows: Dict[str, Sequence[Sequence[float]]], context: Context) -> List[float]:
    """The state the encoding in force writes for one decision's materials, given as rows table by table."""
    if layer == "tactics":
        squad = _squad(rows["squad"][0])
        losses, killed, now = rows["fight"][0]
        target = [_region(row) for row in rows["target"]]
        view = WorldView(observation=None, catalogue=context.catalogue, regions=target)
        members = [_sighting(_unit(row), context.catalogue) for row in rows["members"]]
        threats = [_sighting(_unit(row), context.catalogue) for row in rows["threats"]]
        return tactical_state(squad, members, threats, float(losses), float(killed), view, int(now), context.combat)
    if layer == "operations":
        board = _named(rows["board"][0], BOARD_COLUMNS)
        observation = Observation(frame=0, game_time_ms=int(board["now"]), episode=0, blocks=0, slot=0,
                                  credits=float(board["credits"]), income=float(board["income"]),
                                  units=int(board["units"]), unit_cap=int(board["unit_cap"]),
                                  under_construction=int(board["under_construction"]), killed_units=0,
                                  killed_buildings=0, lost_units=0, lost_buildings=0,
                                  unit_states=[_unit(row) for row in rows["units"]])
        home = int(board["home"])
        view = build(observation, context.catalogue, home if home >= 0 else None, [_region(r) for r in rows["regions"]])
        members: Dict[int, List[int]] = {}
        for squad_row, unit in rows["members"]:
            members.setdefault(int(squad_row), []).append(int(unit))
        squads = [_squad(row, members.get(index, ())) for index, row in enumerate(rows["squads"])]
        orders = None
        if board["orders"]:
            orders = OperationsOrders(posture=Posture(int(board["posture"])),
                                      priorities={int(r): float(p) for r, p in rows["priorities"]},
                                      offensive=bool(board["offensive"]), loss_allowance=float(board["allowance"]),
                                      expansion=[int(row[0]) for row in rows["expansion"]])
        decided = next(s for s in squads if s.id == int(board["squad"]))
        return operational_state(view, orders, squads, int(board["now"]), tuple(int(r[0]) for r in rows["spawns"]),
                                 squad=decided, combat=context.combat, present=forces(view, squads),
                                 access=_access(board, rows))
    if layer == "economy":
        values = _named(rows["board"][0], ECONOMIC_BOARD_COLUMNS)
        board = EconomicBoard(**{name: (Posture(int(v)) if name == "posture" else bool(v) if name in _BOARD_BOOLS
                                        else int(v) if name in _BOARD_INTS else float(v))
                                 for name, v in values.items()},
                              target_mix={Role(int(role)): float(share) for role, share in rows["mix"]})
        slots: List[Optional[Offer]] = [None] * INVESTMENT_SLOTS
        for row in rows["offers"]:
            v = _named(row, OFFER_COLUMNS)
            slot = int(v.pop("slot"))
            slots[slot] = Offer(**{name: (_OFFER_ENUMS[name](int(x)) if name in _OFFER_ENUMS
                                                        else bool(x) if name in _OFFER_BOOLS
                                                        else int(x) if name in _OFFER_INTS else float(x))
                                                 for name, x in v.items()})
        return economic_state(board, slots)
    raise ValueError(f"no layer named {layer!r}")
