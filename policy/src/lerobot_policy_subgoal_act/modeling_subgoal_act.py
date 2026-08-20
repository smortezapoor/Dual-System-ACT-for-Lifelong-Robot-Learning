"""The dual-system policy: a small planner (System 2) steering ACT (System 1).

    System 2  (slow, thinks about the task)      System 1  (fast, moves the arm)
    ---------------------------------------      -----------------------------
    picture + arm state + which task             ACT, conditioned on ONE number
    -> which step the episode is on (0..3)       -> the next 100 arm actions
    about 162k weights                           about 51.6M weights

THE INTERFACE IS ONE NUMBER
---------------------------
System 2 outputs a "phase": how far through the current task the episode is.
The policy turns that into a "skill" id (see common.py for why these are
different) and System 1 looks that up in a small embedding table.

Why I made the interface one number, rather than text or a latent vector:

  - It is cheap enough to recompute every single control step, which is what
    makes this a closed loop instead of a plan made once at the start.
  - It can be swapped at evaluation time without retraining, which is what
    gives me six ablation conditions from three training runs.
  - It can be drawn on a video frame, so the rollout recordings actually show
    what System 2 was thinking.

Being honest about the cost: a sub-goal outside my list of 25 skills simply
cannot be said.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.utils.constants import OBS_IMAGES, OBS_STATE

from .configuration_subgoal_act import SubgoalACTConfig
from .subgoal_index import SubgoalIndex


class SubgoalEmbedding(nn.Module):
    """Adds the sub-goal to the first token ACT builds.

    HOW THE SUB-GOAL REACHES ACT WITHOUT COPYING ACT

    ACT builds its list of encoder tokens inside its own forward() method.
    Adding a genuinely new token would mean copying that whole method into this
    repo, which is the thing the LeRobot docs warn against, because such a copy
    then silently goes stale when LeRobot updates.

    So I wrap the smallest piece that actually has to change: the little Linear
    layer that makes ACT's FIRST encoder token. It does its normal job, and the
    sub-goal embedding is added on top.

    This is not a compromise at rollout time. ACT feeds a zero vector into that
    layer when it is not training, so at rollout the first token carries the
    sub-goal and nothing else.

    Being fair about the downside: adding is a weaker way to condition than
    FiLM, which would let the sub-goal scale the vision features instead of
    just shifting one token. My test for whether adding is enough is the
    A_shuffled condition: if feeding wrong sub-goals did NOT hurt, System 1
    would be ignoring the channel. Measured, it hurts a lot (50.7% -> 34.0%),
    so adding is enough here.
    """

    def __init__(self, act_linear, n_rows, embed_dim, token_dim):
        super().__init__()
        self.act_linear = act_linear                    # ACT's original layer
        self.embedding = nn.Embedding(n_rows, embed_dim)   # 26 rows: 25 skills + "none"
        self.to_token = nn.Linear(embed_dim, token_dim)    # match ACT's token width

        # The policy writes the current skills here just before it runs the
        # model, because the call to this layer happens inside ACT's own
        # forward(), where there is no way to pass an extra argument in.
        self.current_skills = None

    def forward(self, latent):
        token = self.act_linear(latent)
        if self.current_skills is None:
            # Nobody set a sub-goal, so behave exactly like plain ACT.
            return token
        skills = self.current_skills.to(token.device)
        return token + self.to_token(self.embedding(skills))


class Selector(nn.Module):
    """System 2: looks at the scene and says which phase the episode is on.

    I kept it deliberately small. It runs every control step, over hundreds of
    episodes and six conditions, so every weight here gets paid for many times.
    It also reuses the picture features ACT has already computed, instead of
    running a second vision network, which makes it almost free.

    It has NO memory and nothing forcing it to move forwards. That is on
    purpose: if the robot drops the bowl, the right answer really is to go back
    to "grasp the bowl". Those backwards steps are how I measure error recovery,
    so smoothing them away would delete the thing this experiment exists to
    observe.
    """

    def __init__(self, image_feature_dim, state_dim, n_tasks, max_phases):
        super().__init__()
        self.state_net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
        )
        # The task id is handed to it. Only the progress has to be worked out.
        self.task_embedding = nn.Embedding(n_tasks, 32)
        self.head = nn.Sequential(
            nn.Linear(image_feature_dim + 64 + 32, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, max_phases),
        )

    def forward(self, image_features, state, task_ids):
        parts = [image_features, self.state_net(state), self.task_embedding(task_ids)]
        return self.head(torch.cat(parts, dim=-1))


class SubgoalACTPolicy(ACTPolicy):
    """ACT, conditioned on a sub-goal that is re-chosen every control step."""

    config_class = SubgoalACTConfig
    name = "subgoal_act"

    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        self.config = config

        self.index = None
        if config.subgoals_path:
            self.index = SubgoalIndex.load(config.subgoals_path)
            if self.index.n_skills != config.n_skills:
                raise ValueError(
                    "subgoals.json has %d skills but the config says %d. The embedding "
                    "table would be the wrong size."
                    % (self.index.n_skills, config.n_skills)
                )

        # Wrap ACT's latent projection. Done after super().__init__() so that
        # the original layer already exists.
        self.model.encoder_latent_input_proj = SubgoalEmbedding(
            self.model.encoder_latent_input_proj,
            n_rows=config.n_skills + 1,
            embed_dim=config.skill_embed_dim,
            token_dim=config.dim_model,
        )

        state_dim = config.robot_state_feature.shape[0] if config.robot_state_feature else 0
        n_tasks = len(self.index.tasks) if self.index else 10
        self.selector = Selector(config.dim_model, state_dim, n_tasks, config.max_phases)

        self.labels = None          # loaded lazily from the parquet file
        self.step = 0               # step counter, for the "frozen" clock
        self.last_skills = None     # to notice when the sub-goal changes
        self.random = torch.Generator().manual_seed(config.shuffled_seed)

        # Set from outside for eval_mode="oracle". The normal evaluation
        # interface only provides camera images, never simulator state, so the
        # oracle condition has to be driven by my own rollout loop.
        self.oracle_phase = None

    # -- helpers -------------------------------------------------------------
    def get_task_ids(self, batch, batch_size, device):
        """Work out which task each item is, from its instruction text."""
        instructions = batch.get("task")
        if instructions is None or self.index is None:
            return torch.zeros(batch_size, dtype=torch.long, device=device)
        if isinstance(instructions, str):
            instructions = [instructions] * batch_size

        task_ids = []
        for text in instructions:
            task_id = self.index.task_from_instruction(text)
            if task_id is None:
                raise KeyError(
                    "instruction not found in subgoals.json: %r. Without the task id "
                    "the wrong sub-goal would be selected." % text
                )
            task_ids.append(task_id)
        return torch.tensor(task_ids, dtype=torch.long, device=device)

    def get_image_features(self, batch):
        """Reuse ACT's vision backbone, so System 2 costs almost nothing."""
        # The scene camera. The wrist camera mostly shows gripper and table,
        # which is poor evidence for "how far through the task am I".
        image = batch[OBS_IMAGES][0]
        features = self.model.backbone(image)["feature_map"]
        features = self.model.encoder_img_feat_input_proj(features)
        return features.mean(dim=(2, 3))      # average over height and width

    def choose_phases(self, batch, task_ids):
        """One phase per item in the batch, according to eval_mode."""
        config = self.config
        device = task_ids.device
        batch_size = len(task_ids)

        # How many phases each item's task has. Tasks differ, so this is a list.
        if self.index is not None:
            counts = [self.index.n_phases(int(t)) for t in task_ids]
        else:
            counts = [config.max_phases] * batch_size
        n_phases = torch.tensor(counts, device=device)

        mode = config.eval_mode

        if mode == "static":
            # Always this task's first sub-goal. The naive baseline.
            return torch.zeros(batch_size, dtype=torch.long, device=device)

        if mode == "none":
            # One past the last real phase, which becomes "no sub-goal".
            return n_phases

        if mode == "frozen":
            # Decided by the clock, not by what is happening. It cannot go
            # backwards, which makes it a clean control for error recovery.
            fraction = self.step / config.max_episode_steps
            phases = (n_phases * fraction).long()
            return torch.minimum(phases, n_phases - 1)

        if mode == "shuffled":
            # Deliberately wrong, to check System 1 is really listening.
            draws = torch.rand(batch_size, generator=self.random).to(device)
            phases = (draws * n_phases).long()
            return torch.minimum(phases, n_phases - 1)

        if mode == "oracle":
            if self.oracle_phase is None:
                raise RuntimeError(
                    "eval_mode='oracle' needs the true progress, which the normal "
                    "evaluation interface does not provide. Run it from the rollout "
                    "script, which sets policy.oracle_phase every step."
                )
            return torch.full((batch_size,), int(self.oracle_phase),
                              dtype=torch.long, device=device)

        # "learned": System 2 predicts it from what it can see.
        logits = self.selector(self.get_image_features(batch), batch[OBS_STATE], task_ids)

        # Hide the phases this task does not have, then take the best of what
        # is left. Hiding rather than clamping is deliberate: clamping a too-big
        # answer down to the last valid phase would invent a confident answer
        # the model never actually gave.
        all_phases = torch.arange(config.max_phases, device=device)
        valid = all_phases.unsqueeze(0) < n_phases.unsqueeze(1)
        logits = logits.masked_fill(~valid, float("-inf"))
        return logits.argmax(dim=-1)

    def phases_to_skills(self, task_ids, phases):
        """(task, phase) -> skill id, one per batch item."""
        if self.index is None:
            return torch.clamp(phases, max=self.config.no_subgoal_index)
        skills = []
        for task_id, phase in zip(task_ids.tolist(), phases.tolist()):
            skills.append(self.index.skill(task_id, phase))
        return torch.tensor(skills, dtype=torch.long, device=task_ids.device)

    # -- running the policy --------------------------------------------------
    def reset(self):
        super().reset()
        self.step = 0                # the frozen clock restarts each episode
        self.last_skills = None
        self.oracle_phase = None

    @torch.no_grad()
    def select_action(self, batch):
        if self.config.train_mode == "none":
            self.step += 1
            return super().select_action(batch)

        prepared = dict(batch)
        if self.config.image_features:
            prepared[OBS_IMAGES] = [batch[key] for key in self.config.image_features]

        batch_size = len(prepared[OBS_STATE])
        device = prepared[OBS_STATE].device
        task_ids = self.get_task_ids(prepared, batch_size, device)
        phases = self.choose_phases(prepared, task_ids)
        skills = self.phases_to_skills(task_ids, phases)

        # System 2 runs every step, even in the middle of an action chunk. If it
        # changes its mind, the leftover actions are thrown away immediately.
        #
        # This is what makes the loop actually closed. Without it, a new
        # sub-goal only takes effect when the next chunk is requested, so
        # n_action_steps alone would decide how responsive the system is.
        if self.config.flush_on_change:
            changed = self.last_skills is None or bool((skills != self.last_skills).any())
            if changed:
                if hasattr(self, "_action_queue"):
                    self._action_queue.clear()
                self.last_skills = skills.clone()

        self.model.encoder_latent_input_proj.current_skills = skills
        try:
            action = super().select_action(batch)
        finally:
            # Always clear it, so it can never leak into an unrelated forward.
            self.model.encoder_latent_input_proj.current_skills = None

        self.step += 1
        # Read by the video script, to draw what System 2 chose on each frame.
        self.last_choice = {"phase": int(phases[0]), "skill": int(skills[0])}
        return action

    # -- training ------------------------------------------------------------
    def load_labels(self):
        """Read the per-frame phase labels once, and keep them in memory."""
        import pandas as pd

        table = pd.read_parquet(self.config.labels_path).set_index("index")
        self.labels = {
            "phase": table["phase"].to_dict(),
            "task": table["task_id"].to_dict(),
        }

    def get_training_targets(self, batch):
        """The true (skills, phases, task_ids) for this batch of frames."""
        config = self.config
        if config.train_mode == "none" or self.index is None or not config.labels_path:
            return None, None, None
        if "index" not in batch:
            return None, None, None
        if self.labels is None:
            self.load_labels()

        device = batch[OBS_STATE].device
        frame_ids = batch["index"].tolist()

        phases = []
        task_ids = []
        for frame_id in frame_ids:
            task_ids.append(self.labels["task"].get(frame_id, -1))
            if config.train_mode == "static":
                # The naive baseline is trained properly, on a constant first
                # sub-goal, rather than faked at eval time. Feeding a constant
                # to a policy trained on changing sub-goals would be unlike
                # anything it saw in training, and would flatter the full
                # system unfairly.
                phases.append(0 if frame_id in self.labels["phase"] else -1)
            else:
                phases.append(self.labels["phase"].get(frame_id, -1))

        skills = []
        for task_id, phase in zip(task_ids, phases):
            if task_id < 0 or phase < 0:
                skills.append(config.no_subgoal_index)     # frame with no label
            else:
                skills.append(self.index.skill(task_id, phase))

        skills = torch.tensor(skills, dtype=torch.long, device=device)
        phases = torch.tensor(phases, dtype=torch.long, device=device)
        task_ids = torch.tensor(task_ids, dtype=torch.long, device=device)

        # Sometimes replace the real sub-goal with "none", so that row gets
        # trained too. See the note on no_subgoal_prob in the config.
        if config.no_subgoal_prob > 0:
            drop = torch.rand(skills.shape, device=device) < config.no_subgoal_prob
            skills = torch.where(drop, torch.full_like(skills, config.no_subgoal_index), skills)

        return skills, phases, task_ids

    def forward(self, batch):
        skills, phases, task_ids = self.get_training_targets(batch)

        if skills is not None:
            self.model.encoder_latent_input_proj.current_skills = skills
        try:
            # System 1 trains on the TRUE sub-goal, not System 2's guess.
            #
            # This keeps an untrained, near-random System 2 from feeding
            # nonsense to System 1 early in training, and it keeps the two
            # losses independent, so a later failure can be blamed on one of
            # them. The cost, and I state it in the report: at evaluation time
            # System 1 sees System 2's imperfect guesses for the first time, so
            # part of any gap between A_oracle and A_learned is that, rather
            # than System 2 being bad.
            loss, loss_parts = super().forward(batch)
        finally:
            self.model.encoder_latent_input_proj.current_skills = None

        # Train System 2 on the labelled frames.
        if phases is not None and self.config.train_mode == "learned":
            labelled = phases >= 0
            if labelled.any():
                prepared = dict(batch)
                if self.config.image_features:
                    prepared[OBS_IMAGES] = [batch[key] for key in self.config.image_features]

                logits = self.selector(
                    self.get_image_features(prepared)[labelled],
                    prepared[OBS_STATE][labelled],
                    task_ids[labelled],
                )
                selector_loss = F.cross_entropy(logits, phases[labelled])
                loss = loss + self.config.selector_loss_weight * selector_loss

                correct = (logits.argmax(dim=-1) == phases[labelled]).float().mean()
                loss_parts["selector_loss"] = selector_loss.item()
                loss_parts["selector_accuracy"] = correct.item()

        return loss, loss_parts
