"""Settings for the sub-goal conditioned ACT policy.

This inherits every ACT setting (image size, chunk size, learning rate, and so
on) and adds the handful I need for the sub-goal interface.
"""

from dataclasses import dataclass

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.act.configuration_act import ACTConfig

# How System 1 is trained. This is baked into the weights, so each value needs
# its own training run.
#   learned  System 2 picks the sub-goal, and is trained at the same time
#   static   always this task's FIRST sub-goal, for the whole episode
#   none     no sub-goal at all, the plain-ACT floor
TRAIN_MODES = ("learned", "static", "none")

# Where the sub-goal comes from at evaluation time. This is NOT baked in, so it
# can be swapped on an already trained policy. That is what lets one training run
# produce several of my ablation conditions.
#   learned   System 2 predicts it
#   oracle    read the true progress out of the simulator (needs a special loop)
#   frozen    follow a fixed clock, ignoring what is happening
#   shuffled  pick at random, on purpose, as a control
#   static    always this task's first sub-goal
#   none      feed the "no sub-goal" value
EVAL_MODES = ("learned", "oracle", "frozen", "shuffled", "static", "none")


@PreTrainedConfig.register_subclass("subgoal_act")
@dataclass
class SubgoalACTConfig(ACTConfig):
    # --- the interface ---
    # How many distinct skills there are across all 10 tasks. The embedding
    # table gets one extra row on top of this, for "no sub-goal given".
    n_skills: int = 25
    skill_embed_dim: int = 64

    # How many sub-goals the biggest task has. This is System 2's output size:
    # it predicts a phase within the current task, not one of the 25 skills,
    # because the task is already known.
    max_phases: int = 4

    train_mode: str = "learned"
    eval_mode: str = "learned"

    # Where to find subgoals.json and the per-frame labels.
    subgoals_path: str = None
    labels_path: str = None

    # How much the System 2 training loss counts, next to ACT's own loss.
    selector_loss_weight: float = 0.5

    # During training, sometimes feed "no sub-goal" instead of the real one.
    #
    # This matters more than it looks. The "no sub-goal" row is only ever used
    # at evaluation time, by my C_none condition. If it never showed up during
    # training it would keep its random starting values, and that whole
    # condition would be measuring noise instead of an architecture.
    no_subgoal_prob: float = 0.1

    # Used only by the "frozen" clock schedule: how long an episode can run.
    max_episode_steps: int = 520

    # Throw away the queued actions when the sub-goal changes. See the note in
    # modeling_subgoal_act.py; without this, a new sub-goal can wait up to
    # n_action_steps before it affects anything.
    flush_on_change: bool = True

    # Seed for the "shuffled" control, so the random choice is reproducible.
    shuffled_seed: int = 0

    def __post_init__(self):
        super().__post_init__()
        if self.train_mode not in TRAIN_MODES:
            raise ValueError("train_mode must be one of %s, got %r" % (TRAIN_MODES, self.train_mode))
        if self.eval_mode not in EVAL_MODES:
            raise ValueError("eval_mode must be one of %s, got %r" % (EVAL_MODES, self.eval_mode))
        if self.n_skills < 1:
            raise ValueError("n_skills must be positive, got %d" % self.n_skills)

        # A policy trained with no conditioning has no conditioning pathway to
        # use, so asking it for one at eval time cannot work. Catch it here,
        # rather than deep inside a rollout hours later.
        if self.train_mode == "none" and self.eval_mode != "none":
            raise ValueError(
                "this policy was trained with train_mode='none', so it has no "
                "conditioning to switch on. Use eval_mode='none'."
            )

    @property
    def no_subgoal_index(self):
        """The embedding row meaning 'no sub-goal given': one past the last skill."""
        return self.n_skills
