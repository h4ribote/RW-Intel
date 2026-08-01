"""The networks, and how small they are obliged to be.

The tactical layer is asked for a decision five times a second per squad, and eight game instances at ten times speed put that at four hundred batches a second on one six gigabyte card. That budget, and not any view about what architecture suits a real-time strategy game, is what fixes the size here: two hidden layers of sixty-four for the tactical policy is what fits, and the design says so in advance. The operational layer runs at a tenth of the rate and reads the whole board, so it is allowed to be wider — but not deeper, because it is the same card.

All three are actor-critic in one body with two heads, the operational one's action head being the only one split in two (a region and a task). Sharing the trunk is what makes the value estimate cost nothing extra, which matters at this rate, and the value head exists at all because the advantage estimator needs it; nothing else reads it.

The operational head is factorised into a region and a task rather than emitting the 144 combinations, because the two questions are different — where is worth going, and what to do on arrival — and because the legal set is a product of two small masks rather than a sparse subset of a large one. Masking is applied as an additive floor on the logits rather than by renormalising afterwards, so that an illegal action has no gradient at all rather than a vanishing one.

Both carry their own feature list among their parameters, which is the one thing in this file that is not about arithmetic. A set of parameters is only meaningful beside the encoding it was fitted to, and the widths cannot say what that was: rename a feature, or replace one with another of the same shape, and every width stays where it is while two slots have quietly changed meaning. Such a file loads without complaint, trains without complaint, and fights on a reading of the board that is wrong in a way nothing downstream can see. Carrying the list inside the state dictionary means it is saved, loaded and copied by the ordinary means, and every loader can refuse a file that was fitted to a different one.

A file that states no list at all is a different case from a file that states the wrong one, and treating the two alike throws away parameters that are known to be good. It cannot prove what it was fitted to — that is the whole point of writing the list down — but a person who knows that a layer's encoding has not moved since those parameters were fitted knows something true that the file does not say. So there is a way of saying it: an avowal, which writes into the file the list a person swears it was fitted to and their words for why in the same breath. Deliberately into the file rather than onto the command line of the run that loads it, because a flag is a claim made once and never seen again, while this one is saved and copied with the parameters and every loader says out loud that the list beside them is somebody's word and not a fit's record. What an avowal may never do is contradict: it may add a claim the file does not make and never overwrite one it does, since overwriting a claim is exactly the failure writing claims down was for.

And the list is not the whole claim. A slot's name says what it is called and not what it holds, and the day the strategic cut changed from comparing this side's army with the enemy's whole side to comparing armies with armies, the name and the list were untouched — so parameters fitted the day before loaded in silence onto a quantity that had moved. What rides beside the list is therefore a digest of the code that fills the slots, so that a change of meaning retires the parameters fitted to the old meaning exactly as a change of name does. Its own limits are stated where it is computed.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import torch
from torch import nn

from .encoding import (
    OPERATIONAL_FEATURES,
    OPERATIONAL_RECIPE,
    OPERATIONAL_REGIONS,
    OPERATIONAL_SIZE,
    OPERATIONAL_TASKS,
    SQUAD_SLOTS,
    STRATEGIC_ACTIONS,
    STRATEGIC_FEATURES,
    STRATEGIC_RECIPE,
    STRATEGIC_SIZE,
    TACTICAL_ACTIONS,
    TACTICAL_FEATURES,
    TACTICAL_RECIPE,
    TACTICAL_SIZE,
)

#: What a masked-out action's logit is pushed to. Large enough that it never survives a softmax at single precision, finite so that it never produces a not-a-number when every action in a row happens to be masked.
MASKED = -1e9

#: What a network's own feature list is called inside its parameters file. A buffer rather than anything alongside, so that it is saved and loaded by the ordinary means and cannot be separated from the parameters it describes: a list kept in a second file is a list somebody copies the parameters without.
ENCODING_KEY = "encoding"

#: Where a person's words go when they swear what a file with no list of its own was fitted to. A plain entry in the state dictionary and not a buffer on any network, because no network has anything to say here: it is written by a person about parameters that already exist, and it is removed again before the parameters are loaded so that the network sees only what it owns.
AVOWAL_KEY = "encoding_avowed"

#: The first thing a set of parameters multiplies its input by, and so where the width it reads is written down. Named here rather than spelled out wherever a file is inspected, because it is the only part of a state dictionary anything reads without loading it.
INPUT_WEIGHT = "body.0.weight"


class EncodingRefused(ValueError):
    """Parameters that cannot be shown to read the features the network reading them now reads.

    Raised rather than warned about, in the two cases that differ: a file that states a different feature list, which is refused for good, and a file that states none, which is refused until a person avows what it was fitted to. Writing that avowal raises it too, when what is offered is another layer's parameters or a claim with no reason attached to it. A ValueError so that a caller which already refuses unusable parameters that way catches it without knowing this module exists.
    """


def _bytes(text: str) -> torch.Tensor:
    return torch.tensor(list(text.encode("utf-8")), dtype=torch.uint8)


def _text(row: torch.Tensor) -> str:
    return bytes(row.detach().to("cpu").flatten().tolist()).decode("utf-8")


#: What separates the two claims a stamp makes: the list of what the slots are called, and the digest of the code that fills them. A blank line, so that a stamp reads as two paragraphs and a stamp written before there were two still parses as the first alone.
STAMP_BREAK = "\n\n"


def encoding_stamp(features: Sequence[str], recipe: str) -> torch.Tensor:
    """What a set of parameters was fitted to, as a row of bytes which rides inside a state dictionary: the names of the slots, and a digest of the code that fills them.

    Written down at all because the width of the input layer is not evidence about what the features MEAN. Renaming a feature, or replacing one with another of the same shape, leaves every width where it was, so a file fitted to the old meaning loads without complaint and the network then reads two slots as something they no longer are — which is the silent failure this module's neighbours exist to refuse, and the only one they could not see.

    The names alone turned out not to be enough, and the case that showed it is worth keeping. The strategic cut compared this side's fighting strength with the enemy's whole side, buildings and builders included, and the day it was corrected to fighters against fighters the slot was still called `military_edge` and the list was still identical byte for byte. Parameters fitted the day before loaded in silence onto a quantity that had moved beneath them, and the imitation the strategic layer was bootstrapped from was one of them. So the recipe rides beside the names: change what a slot holds and the parameters fitted to the old holding are refused, exactly as they are when the slot is renamed.
    """
    return _bytes("\n".join(features) + STAMP_BREAK + recipe)


def encoding_features(stamp: torch.Tensor) -> Tuple[str, ...]:
    """The feature list back out of a stamp, for saying which feature has moved rather than only that something has."""
    return tuple(_text(stamp).split(STAMP_BREAK)[0].split("\n"))


def encoding_recipe(stamp: torch.Tensor) -> Optional[str]:
    """The digest of the code that filled the slots, or nothing where the stamp was written before a stamp carried one.

    Nothing and not the empty string, because the two are different findings and one of the two is refusable: a file that states a recipe which disagrees is refused for good, and a file that states none is refused until a person avows what it was fitted to — the same shape of answer the feature list itself already gives.
    """
    parts = _text(stamp).split(STAMP_BREAK)
    return parts[1] if len(parts) > 1 and parts[1] else None


def encoding_avowal(state) -> Optional[str]:
    """The words a person wrote when they swore what a set of parameters was fitted to, or nothing where the list beside them was recorded by the fit that produced them.

    Every loader asks, and every loader says what it finds. A list a program wrote down as it fitted the parameters and a list a person wrote down afterwards are both the same bytes by the time they are read, and only this tells them apart — so a run made on somebody's word is reported as one, rather than being indistinguishable from a run made on a file that could prove what it was.
    """
    if not isinstance(state, dict) or AVOWAL_KEY not in state:
        return None
    return _text(state[AVOWAL_KEY])


def reads(state) -> Optional[int]:
    """How many numbers a set of parameters takes in at its first layer, or nothing where there is no first layer to ask.

    The one thing about a file that can be established instead of believed, and it is worth being exact about what it establishes. It says which layer's parameters these are, because the two layers read widths nothing like each other's and a file of the wrong width cannot be loaded into a network at all. It says nothing whatever about what the numbers in those slots mean: rename a feature, or swap one for another of the same shape, and the width sits exactly where it was. So it is enough to refuse a file offered as a layer it is not, and it is never enough to accept one as fitted to the list now in force.
    """
    weight = state.get(INPUT_WEIGHT) if isinstance(state, dict) else None
    if weight is None or not hasattr(weight, "dim") or weight.dim() != 2:
        return None
    return int(weight.shape[1])


def encoding_complaint(state, net: nn.Module) -> Optional[str]:
    """What is wrong with the feature list a parameters file was written under, in words, or nothing at all when it is the list this network reads.

    The network is asked rather than the layer named, because the network is the authority on what it reads and every caller has one in hand by the time it loads anything.

    A file carrying no list at all is refused rather than trusted. It was written before the list was recorded, so nothing in it says which features it was fitted to, and the alternative to refusing it is to assume the answer — which is exactly the assumption that put parameters fitted to one meaning of a slot underneath a run that read another. What the refusal names is the way back: a person who knows the layer's encoding has not moved since those parameters were fitted can avow it into the file, after which the file states a list like any other and is read here like any other.

    The list is quoted by its entries rather than by its slots, and a network says which of the two an entry is. The tactical list names one feature per number of the state; the operational list names each repeated block once, so its few dozen entries describe a four-hundred-wide state and calling their number a count of features would misdescribe the file by an order of magnitude.
    """
    if not isinstance(state, dict) or ENCODING_KEY not in state:
        return ("it carries no feature list at all, so nothing in it says which encoding it was fitted to; "
                "parameters written before the list was recorded cannot be shown to read the features now in "
                "force, so either fit them again or, if you know the layer's encoding has not moved since they "
                "were fitted, say so into the file itself with 'python -m rwintel.learn avow'")
    was = encoding_features(state[ENCODING_KEY])
    now = encoding_features(getattr(net, ENCODING_KEY))
    if was != now:
        for index, (before, after) in enumerate(zip(was, now)):
            if before != after:
                return ("it was fitted to a different feature list: its %s %d is %r where this layer now reads %r"
                        % (net.ENTRY, index, before, after))
        return ("it was fitted to a different feature list: %d %ss where this layer's list has %d"
                % (len(was), net.ENTRY, len(now)))
    made = encoding_recipe(state[ENCODING_KEY])
    makes = encoding_recipe(getattr(net, ENCODING_KEY))
    if made is None:
        return ("it states the feature list it was fitted to but not the recipe those features were made by, so "
                "it was written before a stamp said what a slot HOLDS as well as what it is called; a slot whose "
                "meaning moved under an unchanged name is exactly what the list cannot see, so either fit them "
                "again or, if you know this layer's cut has not moved since they were fitted, say so into the "
                "file itself with 'python -m rwintel.learn avow'")
    if made != makes:
        return ("it was fitted by a different recipe: the names of the slots are this layer's, but the code that "
                "fills them digests to %s where this layer's digests to %s, so at least one slot holds a "
                "different quantity than it did when these were fitted" % (made, makes))
    return None


def load_encoded(net: nn.Module, state) -> Optional[str]:
    """Puts a set of parameters into a network, refusing any whose feature list is not the one the network reads, and says whether the list beside them was a person's word.

    One function rather than a check and a load at every call site, because the two must not come apart: a loader that checked and then loaded something else, or loaded and then checked, would be exactly as wrong as one that never checked at all. It also has the only other thing that has to happen here — lifting a person's avowal out of the dictionary before the parameters go in, since it belongs to no network and a strict load would refuse the whole file for carrying it.

    Returns those words where there are any, so that the caller can say them: parameters accepted on somebody's say-so are worth naming every time they are read, and the caller is the only one that knows what it is about to do with them.
    """
    complaint = encoding_complaint(state, net)
    if complaint is not None:
        raise EncodingRefused(complaint)
    net.load_state_dict({name: value for name, value in state.items() if name != AVOWAL_KEY})
    return encoding_avowal(state)


def avowed(state, net: nn.Module, words: str) -> dict:
    """A copy of a parameters file with the feature list a person swears it was fitted to written into it, and their words for why beside it.

    This is the way back for parameters recorded before the list was. Every loader here refuses such a file and is right to: nothing in it says which encoding produced it. But refusing it is not the same as knowing it is wrong, and where a layer's encoding has not moved since the parameters were fitted, a person knows something true that the file does not state. This is how they state it — into the file, so that it is saved, copied and read back with the parameters and no later reader has to be told separately.

    An avowal may add a claim the file does not make. It may never contradict one it does.

    That distinction is what decides the two kinds of file that arrive here. A file with no stamp at all states nothing, and the whole stamp is added. A file that states this layer's feature list but no recipe — written while the stamp carried only names — states half of it, and the half it is missing is added while the half it makes is checked first and must agree. What is refused for good is a file whose list disagrees with this layer's, or whose recipe does: overwriting a claim is precisely the failure writing claims down was for, and a person's word is not evidence against a record the fit itself left.

    A file whose first layer does not read what this network reads cannot be avowed as this network's. That is the whole of what a width proves, and it is exactly what is wanted here: it is what stops one layer's name, pointed at a directory of parameters, from stamping another layer's files with a list they were never fitted to. What a width does not prove is that a file of the right width was fitted to the list now in force — which is why a person's word is required at all.

    And that word may not be empty. Somebody has to have written down why they believe it, because that sentence is the only evidence the file will ever carry for a claim no program can check.
    """
    if not isinstance(state, dict):
        raise EncodingRefused("this is not a set of parameters at all")
    if AVOWAL_KEY in state:
        raise EncodingRefused(
            "it already carries somebody's avowal, and an avowal is a person's word about a file that has not "
            "had one; writing over one is the very thing writing it down was for")
    if ENCODING_KEY in state:
        was = encoding_features(state[ENCODING_KEY])
        now = encoding_features(getattr(net, ENCODING_KEY))
        if was != now:
            raise EncodingRefused(
                "it states a feature list that is not this layer's, and a list that disagrees is refused for "
                "good rather than until somebody runs this: %d %ss against this layer's %d"
                % (len(was), net.ENTRY, len(now)))
        if encoding_recipe(state[ENCODING_KEY]) is not None:
            raise EncodingRefused(
                "it already states both what its slots are called and the recipe they were made by, so there is "
                "nothing here a person can add; a recipe that disagrees with this layer's is refused for good")
    width = reads(state)
    wanted = net.body[0].in_features
    if width != wanted:
        raise EncodingRefused(
            "its first layer reads %s number(s) where this layer reads %d, so these are another layer's "
            "parameters and this layer's feature list would say something false about them"
            % ("none" if width is None else width, wanted))
    if not words.strip():
        raise EncodingRefused(
            "an avowal needs the reason a person believes it, since that sentence is the only evidence the "
            "file will carry for a claim nothing in it can check")
    avowal = dict(state)
    avowal[ENCODING_KEY] = getattr(net, ENCODING_KEY).detach().clone()
    avowal[AVOWAL_KEY] = _bytes(words.strip())
    return avowal


def _trunk(inputs: int, width: int, depth: int = 2) -> nn.Sequential:
    layers: list = []
    size = inputs
    for _ in range(depth):
        layers.append(nn.Linear(size, width))
        layers.append(nn.Tanh())
        size = width
    return nn.Sequential(*layers)


def _initialise(module: nn.Module, gain: float = 1.0) -> nn.Module:
    """Orthogonal weights with a small gain on the output heads, which is what stops a freshly built policy from starting out nearly deterministic. A policy that begins confident explores nothing, and with this sample budget there is no room to wait for it to be argued out of its first opinion."""
    for layer in module.modules():
        if isinstance(layer, nn.Linear):
            nn.init.orthogonal_(layer.weight, gain)
            nn.init.zeros_(layer.bias)
    return module


class TacticalNet(nn.Module):
    """One squad's fight to one of the departures, with a value for it."""

    #: What one entry of this network's feature list stands for, for a refusal that has to quote how many there are. Here it is one number of the state per name, so a count of entries is a count of features and the plain word is the true one.
    ENTRY = "feature"

    def __init__(self, width: int = 64) -> None:
        super().__init__()
        self.body = _initialise(_trunk(TACTICAL_SIZE, width), gain=2.0 ** 0.5)
        self.action = _initialise(nn.Linear(width, TACTICAL_ACTIONS), gain=0.01)
        self.value = _initialise(nn.Linear(width, 1), gain=1.0)
        # The feature list travels with the parameters, because the widths above cannot tell a renamed feature from the one it replaced and a file fitted to the old meaning would otherwise load in silence.
        self.register_buffer(ENCODING_KEY, encoding_stamp(TACTICAL_FEATURES, TACTICAL_RECIPE))

    def forward(self, state: torch.Tensor, mask: Optional[torch.Tensor] = None):
        hidden = self.body(state)
        logits = self.action(hidden)
        if mask is not None:
            logits = logits.masked_fill(mask <= 0, MASKED)
        return logits, self.value(hidden).squeeze(-1)


class OperationalNet(nn.Module):
    """The whole board plus which squad is being decided about, to a region and a task, with a value for the pair.

    The squad is named to the network as a slot rather than by handing it its own row separately, because the row is already in the board: the layer's decision about squad three depends on where squads one and two have been sent, and a network that could not see them would be deciding eight independent problems that are not independent.
    """

    #: What one entry of this network's feature list stands for. NOT one number of the state per name: the operational list names a few aggregates and then each repeated block once, so its few dozen entries describe a four-hundred-wide state, and a refusal that called them features would misdescribe the file by an order of magnitude.
    ENTRY = "block"

    def __init__(self, width: int = 192) -> None:
        super().__init__()
        self.body = _initialise(_trunk(OPERATIONAL_SIZE + SQUAD_SLOTS, width), gain=2.0 ** 0.5)
        self.region = _initialise(nn.Linear(width, OPERATIONAL_REGIONS), gain=0.01)
        self.task = _initialise(nn.Linear(width, OPERATIONAL_TASKS), gain=0.01)
        self.value = _initialise(nn.Linear(width, 1), gain=1.0)
        # Its own feature list, for the reason the tactical network carries one.
        self.register_buffer(ENCODING_KEY, encoding_stamp(OPERATIONAL_FEATURES, OPERATIONAL_RECIPE))

    def forward(self, state: torch.Tensor, squad: torch.Tensor,
                region_mask: Optional[torch.Tensor] = None,
                task_mask: Optional[torch.Tensor] = None):
        hidden = self.body(torch.cat([state, squad], dim=-1))
        regions = self.region(hidden)
        tasks = self.task(hidden)
        if region_mask is not None:
            regions = regions.masked_fill(region_mask <= 0, MASKED)
        if task_mask is not None:
            tasks = tasks.masked_fill(task_mask <= 0, MASKED)
        return regions, tasks, self.value(hidden).squeeze(-1)


class StrategicNet(nn.Module):
    """The match as an aggregate to one of the five postures, with a value for it.

    The smallest of the three and by a wide margin the cheapest to run: one decision every ten seconds for a whole side, against five a second per squad for the tactical layer. Nothing about the throughput budget that fixed the other two widths reaches this one, so the width here is the width of the problem — a few dozen aggregates into a choice of five — and it is kept at the tactical layer's rather than raised, because a wider body over a thirty-wide input with one label per ten seconds of match is a body fitted to the noise in the few thousand decisions a training run can afford.

    Its own feature list rides with its parameters for the reason the other two carry one.
    """

    #: What one entry of this network's feature list stands for: one number of the state per name, as the tactical list is.
    ENTRY = "feature"

    def __init__(self, width: int = 64) -> None:
        super().__init__()
        self.body = _initialise(_trunk(STRATEGIC_SIZE, width), gain=2.0 ** 0.5)
        self.action = _initialise(nn.Linear(width, STRATEGIC_ACTIONS), gain=0.01)
        self.value = _initialise(nn.Linear(width, 1), gain=1.0)
        self.register_buffer(ENCODING_KEY, encoding_stamp(STRATEGIC_FEATURES, STRATEGIC_RECIPE))

    def forward(self, state: torch.Tensor, mask: Optional[torch.Tensor] = None):
        hidden = self.body(state)
        logits = self.action(hidden)
        if mask is not None:
            logits = logits.masked_fill(mask <= 0, MASKED)
        return logits, self.value(hidden).squeeze(-1)


def one_hot_slot(slot: int, device=None) -> torch.Tensor:
    row = torch.zeros(SQUAD_SLOTS, device=device)
    if 0 <= slot < SQUAD_SLOTS:
        row[slot] = 1.0
    return row


def sample(logits: torch.Tensor, greedy: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
    """Draws from a categorical distribution over the logits, or takes its mode. Returns the choice and its log probability, which is what the optimiser needs to know how likely the behaviour policy thought the behaviour was."""
    distribution = torch.distributions.Categorical(logits=logits)
    choice = logits.argmax(dim=-1) if greedy else distribution.sample()
    return choice, distribution.log_prob(choice)


def entropy(logits: torch.Tensor) -> torch.Tensor:
    return torch.distributions.Categorical(logits=logits).entropy()
