"""What answers when a layer is asked to decide.

A decider is the one thing a learnt layer holds that a script layer does not: everything else about the layer -the reports it sends up, the way a contract is priced, the rules that decide when a departure is re-issued -is the same code. That is deliberate and it is what the comparison the whole project rests on requires. If a learnt layer and a script layer differed in more than the decision, then beating the script would not mean the decision had got better.

There is a third possibility and it is why the interface is shaped this way: no decider at all. A learnt layer built without one falls through to the rule it inherited and plays what the rule chose, which turns the script into a teacher producing state and action pairs of exactly the form a learnt layer emits.

Every decider says what distribution it drew its answer from (`Choice.probabilities`), because a recorded decision is only usable for learning off the policy that took it when the probability that policy gave it is known. A rule is a distribution too: one that puts everything on its answer.
"""

from __future__ import annotations

import math
import random
import threading
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence

from .inference import Batcher, limits


@dataclass
class Choice:
    """A decision and what the decider knew about it."""

    action: int
    log_prob: float = 0.0
    value: float = 0.0
    #: The second half of a decision made in two parts, as the operational layer chooses a region and then a task.
    second: int = -1
    second_log_prob: float = 0.0
    #: Carried onto the recorded step unchanged; see `Step.meta`.
    meta: Optional[dict] = None
    #: The distribution the first choice was drawn from, over every action, and the one the second was drawn from given the first that was taken. None from a decider that cannot say, which is a person's play inferred from a replay.
    probabilities: Optional[List[float]] = None
    second_probabilities: Optional[List[float]] = None
    #: The distribution over the second choice for every possible first one, row by row, from a decider that can say; what an exploring decider needs to price a first choice it substituted.
    second_by_first: Optional[List[List[float]]] = None
    #: Which parameters answered: the number of updates the network had been through when it was evaluated, nought from anything that is not a learning network.
    version: int = 0

    @property
    def total_log_prob(self) -> float:
        return self.log_prob + self.second_log_prob


def one_hot(index: int, width: int) -> List[float]:
    return [1.0 if position == index else 0.0 for position in range(width)]


def uniform(mask: Sequence[float]) -> List[float]:
    legal = sum(1 for allowed in mask if allowed > 0)
    return [(1.0 / legal if allowed > 0 else 0.0) for allowed in mask] if legal else [0.0] * len(mask)


def _log(probability: float) -> float:
    return math.log(probability) if probability > 0 else -math.inf


class Labelled:
    """A pupil that plays, now and then substituting an action drawn evenly from the legal ones. The pupil is usually a network, and may be any decider: a rule, a pinned departure, or the script itself (`ScriptChoice`), which is how the script explores around its own answers.

    A teacher's own decisions only cover the boards the teacher's own play leads to. A pupil fitted to them drifts, as soon as it plays for itself, onto boards the teacher never saw, and nothing in the teacher's own record says what to do there. So the pupil plays here, drawing from its distribution and with probability `explore` from all the legal actions evenly; the layer asks the teacher's judge about every board, as it does for every decision, so the record covers the boards the pupil actually visits.

    What was played was drawn from a mixture, and the mixture is what the probabilities say: `(1 - explore)` of the pupil's distribution plus `explore` spread evenly over the legal actions. For a decision made in two parts the pair is drawn as a pair, so the second part's probability is the mixture's, conditioned on the first part that was played.
    """

    def __init__(self, pupil, explore: float = 0.0, seed: int = 0) -> None:
        self.pupil = pupil
        self.explore = explore
        self.random = random.Random(seed)

    def bind(self, judge) -> None:
        bind = getattr(self.pupil, "bind", None)
        if bind is not None:
            bind(judge)

    @property
    def set_input(self) -> bool:
        """Whether the pupil reads the tactical set row rather than the flat features."""
        return getattr(self.pupil, "set_input", False)

    def choose(self, *request) -> Optional[Choice]:
        choice = self.pupil.choose(*request)
        if choice is None:
            return None
        explored = self.random.random() < self.explore
        if len(request) == 2:
            _, mask = request
            first = choice.probabilities or one_hot(choice.action, len(mask))
            mixed = [(1.0 - self.explore) * p + self.explore * q for p, q in zip(first, uniform(mask))]
            action = self.random.choice([i for i, allowed in enumerate(mask) if allowed > 0]) if explored else choice.action
            return Choice(action=action, log_prob=_log(mixed[action]), value=choice.value, probabilities=mixed,
                          version=choice.version, meta=choice.meta)
        _, _, region_mask, plan_masks = request
        regions = uniform(region_mask)
        width = len(plan_masks[0]) if plan_masks else 0
        first = choice.probabilities or one_hot(choice.action, len(region_mask))
        table = choice.second_by_first or [one_hot(choice.second, width) for _ in region_mask]
        if explored:
            region = self.random.choice([i for i, allowed in enumerate(region_mask) if allowed > 0])
            plan = self.random.choice([i for i, allowed in enumerate(plan_masks[region]) if allowed > 0])
        else:
            region, plan = choice.action, choice.second
        plans = uniform(plan_masks[region])
        mixed = [(1.0 - self.explore) * p + self.explore * q for p, q in zip(first, regions)]
        joint = [(1.0 - self.explore) * first[region] * p + self.explore * regions[region] * q
                 for p, q in zip(table[region], plans)]
        given = [p / mixed[region] for p in joint] if mixed[region] > 0 else list(plans)
        return Choice(action=region, second=plan, log_prob=_log(mixed[region]), second_log_prob=_log(given[plan]),
                      value=choice.value, probabilities=mixed, second_probabilities=given, version=choice.version,
                      meta=choice.meta)


class ScriptChoice:
    """Answers with the layer's own judge, which is the script's decision put in the form of a decider, so that the script can play inside `Labelled` and explore around its own answers.

    It holds no judge of its own: the learnt layer it is handed to binds its judge (`bind`), so what plays is the very rule ladder that labels the record.
    """

    def __init__(self) -> None:
        self.judge = None

    def bind(self, judge) -> None:
        self.judge = judge

    def choose(self, *request) -> Optional[Choice]:
        if self.judge is None:
            raise RuntimeError("a script choice answers with a layer's judge, and no layer has bound one")
        if len(request) == 2:
            state, mask = request
            action = int(self.judge.choose(state))
            return Choice(action=action, probabilities=one_hot(action, len(mask)))
        state, slot, region_mask, plan_masks = request
        chosen = self.judge.choose(state, slot, region_mask, plan_masks)
        if chosen is None:
            return None
        region, plan = chosen
        return Choice(action=region, second=plan, probabilities=one_hot(region, len(region_mask)),
                      second_probabilities=one_hot(plan, len(plan_masks[region])))


class NetworkChoice:
    """Answers a choice of one thing from a board and a mask -a tactical departure, or the economy's next investment- from a network, through the batching server when there is one."""

    def __init__(self, net, device=None, batcher: Optional[Batcher] = None, greedy: bool = False) -> None:
        self.net = net
        self.device = device
        self.batcher = batcher
        self.greedy = greedy

    @property
    def set_input(self) -> bool:
        """True for a tactical set network, which reads the set row (`tokens.set_state`) that the layer builds from the decision's materials instead of the 78 flat features."""
        return getattr(self.net, "KIND", "flat") == "set" and getattr(self.net, "LAYER", "") == "tactics"

    def choose(self, state: Sequence[float], mask: Sequence[float]) -> Choice:
        if self.batcher is not None:
            return self.batcher.submit((list(state), list(mask)))
        return evaluate_single(self.net, [(list(state), list(mask))], self.device, self.greedy)[0]


class PinnedDeparture:
    """Answers with one departure, whatever it is shown.

    An ablation rather than a policy, and the reason it exists as a decider rather than as a file of parameters is that it is a measurement the arena is read against. What the score of a fight can be moved by at all is bounded below by what the departures are worth, and the way to find that bound is to take them away one at a time: a layer that never departs from its contract leaves the engine's own attack-move to fight the whole fight, and a layer that always breaks off gives up every fight it could have won. Written as a decider it cannot go stale when the action space changes, and it needs no tensor library at all.
    """

    def __init__(self, action: int) -> None:
        self.action = int(action)

    def choose(self, state: Sequence[float], mask: Sequence[float]) -> Choice:
        return Choice(action=self.action, probabilities=one_hot(self.action, len(mask)))


class PinnedOperations:
    """Answers the operational question by one fixed rule read off the encoded board, for measuring how far the operational decision moves a match at all.

    `home` sends every squad to the region nearest home, `nearest` to the nearest region with enemy strength or enemy extractors in it, `weakest` to the hostile region where the enemy stands weakest against our whole army, `richest` to the hostile region with the most enemy extractors, `spawn` to the enemy's starting region, `random` to a region drawn evenly from those on the map, and `carry` to the region `nearest` picks. Ties go to the nearer region, and a board with nothing hostile on it sends squads to the farthest region. The plan is the first one open in that region, which is the first task the squad's doctrine allows by the first means open; `carry` instead takes the first open plan whose means is a lift, and the first open plan when no lift is open there. Like the pinned departures these are ablations rather than policies.
    """

    RULES = ("home", "nearest", "weakest", "richest", "spawn", "random", "carry")

    def __init__(self, rule: str, seed: int = 0) -> None:
        if rule not in self.RULES:
            raise ValueError(f"no operational rule named {rule!r}: expected one of {', '.join(self.RULES)}")
        self.rule = rule
        self.random = random.Random(seed)

    def choose(self, state: Sequence[float], slot: int, region_mask: Sequence[float],
               plan_masks: Sequence[Sequence[float]]) -> Optional[Choice]:
        from ..control.policy.encoding import GLOBAL_SIZE, MEANS, REGION_FEATURES, REGION_SIZE

        live = [index for index, allowed in enumerate(region_mask) if allowed > 0 and any(plan_masks[index])]
        if not live:
            return None

        def feature(region: int, name: str) -> float:
            return state[GLOBAL_SIZE + region * REGION_SIZE + REGION_FEATURES.index(name)]

        def distance(r: int) -> float:
            return feature(r, "distance")

        if self.rule == "random":
            region = self.random.choice(live)
            drawn = uniform(region_mask)
        else:
            if self.rule == "home":
                region = min(live, key=distance)
            elif self.rule == "spawn":
                home = min(live, key=distance)
                starts = [r for r in live if feature(r, "spawn") > 0 and r != home]
                region = max(starts or live, key=distance)
            else:
                hostile = [r for r in live if feature(r, "enemy_present") > 0 or feature(r, "held_by_enemy") > 0]
                if not hostile:
                    region = max(live, key=distance)
                elif self.rule == "weakest":
                    region = min(hostile, key=lambda r: (feature(r, "enemy_present"), distance(r)))
                elif self.rule == "richest":
                    region = min(hostile, key=lambda r: (-feature(r, "held_by_enemy"), distance(r)))
                else:
                    # `nearest`, and `carry`, which chooses its region the same way.
                    region = min(hostile, key=distance)
            drawn = one_hot(region, len(region_mask))
        width = len(plan_masks[region])

        def first_open(row: Sequence[float]) -> int:
            open_plans = [i for i, allowed in enumerate(row) if allowed > 0]
            if self.rule == "carry":
                # A plan index is task * MEANS + means, and means 0 is walking.
                lifted = [i for i in open_plans if i % MEANS != 0]
                if lifted:
                    return lifted[0]
            return open_plans[0] if open_plans else 0

        table = [one_hot(first_open(row), width) for row in plan_masks]
        plan = table[region].index(1.0)
        return Choice(action=region, second=plan, log_prob=_log(drawn[region]), probabilities=drawn,
                      second_probabilities=table[region], second_by_first=table)


class NetworkOperations:
    def __init__(self, net, device=None, batcher: Optional[Batcher] = None, greedy: bool = False) -> None:
        self.net = net
        self.device = device
        self.batcher = batcher
        self.greedy = greedy

    def choose(self, state: Sequence[float], slot: int, region_mask: Sequence[float],
               plan_masks: Sequence[Sequence[float]]) -> Optional[Choice]:
        request = (list(state), int(slot), list(region_mask), [list(row) for row in plan_masks])
        if self.batcher is not None:
            return self.batcher.submit(request)
        return evaluate_operational(self.net, [request], self.device, self.greedy)[0]


def evaluate_single(net, requests: Sequence[tuple], device=None, greedy: bool = False,
                    version: Optional[Callable[[], int]] = None) -> List[Choice]:
    """One forward pass for a whole batch of one-part choices, wherever they came from. Imported lazily so that a control process that is only running scripts never loads the tensor library at all."""
    import torch

    from .net import sample

    states = torch.tensor([request[0] for request in requests], dtype=torch.float32, device=device)
    masks = torch.tensor([request[1] for request in requests], dtype=torch.float32, device=device)
    stamp = version() if version is not None else 0
    with torch.no_grad():
        logits, values = net(states, masks)
        actions, log_probs = sample(logits, greedy)
        rows = _fetch(actions, log_probs, values, torch.softmax(logits, dim=-1))
    return [Choice(action=int(row[0]), log_prob=row[1], value=row[2], probabilities=row[3:], version=stamp)
            for row in rows]


def evaluate_operational(net, requests: Sequence[tuple], device=None, greedy: bool = False,
                         version: Optional[Callable[[], int]] = None) -> List[Choice]:
    import torch

    from ..control.policy.encoding import OPERATIONAL_PLANS as PLANS, OPERATIONAL_REGIONS as REGIONS
    from .net import one_hot_slot, sample

    states = torch.tensor([request[0] for request in requests], dtype=torch.float32, device=device)
    slots = torch.stack([one_hot_slot(request[1], device=device) for request in requests])
    regions = torch.tensor([request[2] for request in requests], dtype=torch.float32, device=device)
    plans = torch.tensor([request[3] for request in requests], dtype=torch.float32, device=device)
    stamp = version() if version is not None else 0
    with torch.no_grad():
        hidden = net.hidden(states, slots)
        region_logits = net.regions(hidden, regions)
        values = net.value(hidden).squeeze(-1)
        chosen_regions, region_log = sample(region_logits, greedy)
        rows_of = torch.arange(hidden.shape[0], device=hidden.device)
        chosen_plans, plan_log = sample(net.plans(hidden, chosen_regions, plans[rows_of, chosen_regions]), greedy)
        # The plan distribution for every region at once, each under its own region's mask, so that a decider substituting another region can price the plan drawn with it.
        count = hidden.shape[0]
        every = torch.arange(REGIONS, device=hidden.device).repeat(count)
        table = torch.softmax(net.plans(hidden.repeat_interleave(REGIONS, dim=0), every,
                                        plans.reshape(count * REGIONS, PLANS)), dim=-1).reshape(count, -1)
        rows = _fetch(chosen_regions, region_log, values, chosen_plans, plan_log,
                      torch.softmax(region_logits, dim=-1), table)
    answers = []
    for row in rows:
        region = int(row[0])
        by_region = [row[5 + REGIONS + r * PLANS:5 + REGIONS + (r + 1) * PLANS] for r in range(REGIONS)]
        answers.append(Choice(action=region, log_prob=row[1], value=row[2], second=int(row[3]), second_log_prob=row[4],
                              probabilities=row[5:5 + REGIONS], second_probabilities=by_region[region],
                              second_by_first=by_region, version=stamp))
    return answers


def _fetch(*tensors) -> List[List[float]]:
    """Brings a batch of answers back from the device in one transfer, one row per request.

    Reading them one number at a time is what an obvious implementation does and it is ruinous: every scalar taken off a device tensor waits for the device to finish, so a batch costs hundreds of round trips. Laying every column side by side first means one wait for the whole batch, whatever its size.
    """
    import torch

    columns = [tensor.float().reshape(tensor.shape[0], -1) for tensor in tensors]
    return torch.cat(columns, dim=1).cpu().tolist()


def _locked(evaluate: Callable, lock: Optional[threading.Lock]) -> Callable:
    """`evaluate` run inside `lock`, the one an optimiser takes while it writes the weights, so that no forward pass reads a half-written update."""
    if lock is None:
        return evaluate

    def run(requests):
        with lock:
            return evaluate(requests)

    return run


def choice_batcher(net, device=None, greedy: bool = False, version: Optional[Callable[[], int]] = None,
                   lock: Optional[threading.Lock] = None, **kwargs) -> Batcher:
    kwargs = {**dict(zip(("window", "max_batch"), limits(device))), **kwargs}
    return Batcher(_locked(lambda requests: evaluate_single(net, requests, device, greedy, version), lock), **kwargs)


def operational_batcher(net, device=None, greedy: bool = False, version: Optional[Callable[[], int]] = None,
                        lock: Optional[threading.Lock] = None, **kwargs) -> Batcher:
    kwargs = {**dict(zip(("window", "max_batch"), limits(device))), **kwargs}
    return Batcher(_locked(lambda requests: evaluate_operational(net, requests, device, greedy, version), lock),
                   **kwargs)
