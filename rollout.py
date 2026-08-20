"""Running a trained policy inside the simulator, step by step.

The normal evaluator (lerobot-eval) is a black box: it gives back a success rate
and nothing else. Two things we need cannot be done through it.

  1. The oracle condition needs the true progress from the simulator, which the
     policy interface never exposes.
  2. Drawing System 2's choice on a video frame needs to know that choice, which
     only the code running the loop can see.

So both the video script and the perturbation study use this instead.
"""

import numpy as np
import torch

# The simulator and the training data use different names for the same cameras.
SCENE_CAMERA = "agentview_image"
WRIST_CAMERA = "robot0_eye_in_hand_image"


def observation_to_state(obs):
    """Build the 8-number state vector the policy was trained on.

    That is: gripper position (3), gripper rotation as an axis-angle (3), and
    the two finger positions (2).

    The rotation conversion is worth being careful about. The simulator gives a
    quaternion. Taking its first three numbers looks like it should work and does
    not: those are scaled by sin(angle / 2), not by the angle itself. The policy
    was trained on the proper axis-angle, so the shortcut quietly feeds it the
    wrong rotation.
    """
    position = np.asarray(obs["robot0_eef_pos"], dtype=np.float32)
    quaternion = np.asarray(obs["robot0_eef_quat"], dtype=np.float32)
    fingers = np.asarray(obs["robot0_gripper_qpos"], dtype=np.float32)

    # quaternion (x, y, z, w) -> axis-angle
    w = float(np.clip(quaternion[3], -1.0, 1.0))
    angle = 2.0 * np.arccos(w)
    sin_half = float(np.sqrt(max(0.0, 1.0 - w * w)))
    if sin_half < 1e-6:
        axis_angle = np.zeros(3, dtype=np.float32)      # no rotation
    else:
        axis = quaternion[:3] / sin_half
        axis_angle = (axis * angle).astype(np.float32)

    return np.concatenate([position, axis_angle, fingers]).astype(np.float32)


class Runner:
    """Loads a trained policy and feeds it simulator observations."""

    def __init__(self, checkpoint, eval_mode="learned", subgoals_path="",
                 n_action_steps=10, device="cuda"):
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.policies.factory import get_policy_class, make_pre_post_processors
        from lerobot.utils.import_utils import register_third_party_plugins

        # LeRobot only looks for out-of-tree policies inside its own train and
        # eval commands, not when you import it. A script like this has to ask,
        # or loading the checkpoint fails with "policy type 'subgoal_act' is not
        # registered" and a list of built-in policies that points nowhere near
        # the real problem.
        register_third_party_plugins()

        config = PreTrainedConfig.from_pretrained(checkpoint)
        config.pretrained_path = checkpoint

        # Override how much of each predicted chunk we run. Every other path
        # does this too. Left at the trained default of 100, the policy commits
        # to 10 seconds of motion per decision, System 2's answer reaches
        # System 1 about five times per episode, and the closed loop we are
        # trying to measure cannot appear at all.
        config.n_action_steps = n_action_steps

        # Set where the sub-goal comes from EXPLICITLY. Left alone it keeps
        # whatever was baked in during training, which is "learned" for the
        # A_learned checkpoint. Every eval-only condition would then quietly be
        # measuring A_learned under a different name.
        if hasattr(config, "eval_mode"):
            config.eval_mode = eval_mode
            if subgoals_path and not config.subgoals_path:
                config.subgoals_path = subgoals_path

        # We do NOT use make_policy() here. It wants dataset statistics or an
        # environment so it can work out shapes for a policy built from scratch.
        # A trained checkpoint already contains all of that, and going through
        # make_policy would download the whole dataset just to throw it away.
        policy_class = get_policy_class(config.type)
        self.policy = policy_class.from_pretrained(checkpoint, config=config).to(device).eval()
        self.pre, self.post = make_pre_post_processors(config, pretrained_path=checkpoint)
        self.device = device
        self.config = config

    def reset(self):
        self.policy.reset()

    def prepare_image(self, image):
        """Simulator image -> the tensor the policy expects.

        The 180-degree rotation is not a guess. LIBERO renders its frames
        rotated compared with the recorded dataset the policy trained on.
        Measured against the dataset at matching start states, rotating both
        axes gives an error of 0.011, flipping only top-to-bottom gives 0.038,
        and doing nothing gives 0.050. In success terms: with only the vertical
        flip the policy scored 0 out of 10, and with both axes it scored 5 of 8.
        """
        rotated = image[::-1, ::-1].copy()
        tensor = torch.from_numpy(rotated).permute(2, 0, 1).float() / 255.0
        return tensor.unsqueeze(0).to(self.device)

    @torch.no_grad()
    def act(self, obs, instruction):
        """One simulator observation -> one action."""
        state = observation_to_state(obs)
        batch = {
            # Renaming the cameras is the dangerous part. Normalisation
            # statistics are stored per name, so a wrong name here does not
            # crash: it just feeds the policy badly scaled images forever.
            "observation.images.image": self.prepare_image(obs[SCENE_CAMERA]),
            "observation.images.image2": self.prepare_image(obs[WRIST_CAMERA]),
            "observation.state": torch.from_numpy(state).unsqueeze(0).to(self.device),
            "task": [instruction],
        }
        action = self.policy.select_action(self.pre(batch))
        return self.post(action).squeeze(0).cpu().numpy()

    def last_choice(self):
        """What System 2 picked on the last step, or None if there is no System 2.

        The C_none condition genuinely has nothing to report, because its
        select_action returns before ever choosing a sub-goal.
        """
        return getattr(self.policy, "last_choice", None)

    def set_oracle_phase(self, phase):
        """Feed the true phase in, for the oracle condition."""
        self.policy.oracle_phase = phase


def count_transitions(phases):
    """How many times the phase moved forwards, and how many times backwards.

    Backwards is the interesting one. It is error recovery made countable: if
    the robot drops the bowl, the right answer really is to go back to "grasp
    the bowl", and a policy that plans once at the start cannot do that.

    We count PHASES, not skill ids. Phases only ever go 0, 1, 2, 3 within a
    task, so "bigger means further along" is true by construction. Skill ids are
    just names and have no order, so comparing them would be meaningless.
    """
    forwards = 0
    backwards = 0
    for before, after in zip(phases, phases[1:]):
        if after > before:
            forwards += 1
        elif after < before:
            backwards += 1
    return forwards, backwards
