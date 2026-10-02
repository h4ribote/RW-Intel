"""Learning the three layers the design says are worth learning.

What is here is the environment, not a result. The command chain is written and measured, the contracts between its layers are fixed and carried on the wire, and evaluation can say whether one policy beats another and how many episodes that claim needs. This package is what turns those into something a policy can be fitted against: an encoding of the board for each layer, a reward for each that is written in terms of what that layer is for, somewhere to keep decisions until there are enough of them, an inference server that answers every connected instance in one call, and an arena in which engagements are built and fought without playing matches to reach them.

Only the three layers the design nominates are learnt, one at a time, against the other four frozen - including the interference of a script intruder, because a policy that has never had a contract broken has no answer when one is.

Nothing here imports the tensor library at module level. A control process that is only running scripts should not pay for it, and the encoding, the reward and the buffers are all useful without it: they are what a recorded run of the script chain produces its teacher data through.
"""

from ..control.policy.encoding import (
    ECONOMIC_SIZE,
    INVESTMENT_SLOTS,
    OPERATIONAL_PLANS,
    OPERATIONAL_REGIONS,
    OPERATIONAL_SIZE,
    OPERATIONAL_TASKS,
    TACTICAL_ACTIONS,
    TACTICAL_SIZE,
    economic_state,
    investment_mask,
    operational_state,
    plan_masks,
    region_mask,
    squad_mask,
    tactical_state,
    task_mask,
)
from .reward import EconomicReward, OperationalReward, Outcome, TacticalReward
from .rollout import Rollout, Step, Trajectory

__all__ = [
    "ECONOMIC_SIZE",
    "EconomicReward",
    "INVESTMENT_SLOTS",
    "OPERATIONAL_PLANS",
    "OPERATIONAL_REGIONS",
    "OPERATIONAL_SIZE",
    "OPERATIONAL_TASKS",
    "OperationalReward",
    "Outcome",
    "Rollout",
    "Step",
    "TACTICAL_ACTIONS",
    "TACTICAL_SIZE",
    "TacticalReward",
    "Trajectory",
    "economic_state",
    "investment_mask",
    "operational_state",
    "plan_masks",
    "region_mask",
    "squad_mask",
    "tactical_state",
    "task_mask",
]
