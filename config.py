"""Every setting that more than one script needs.

Change a value here and it changes everywhere. That is the only way the ablation
conditions stay comparable: if one condition trained at a different batch size,
the comparison would be measuring the batch size.
"""

import os

ROOT = os.path.dirname(os.path.abspath(__file__))

# Where results go. Overridable so Docker can point it at a mounted folder.
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", os.path.join(ROOT, "outputs"))
CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, "checkpoints")
EVAL_DIR = os.path.join(OUTPUT_DIR, "eval")
VIDEO_DIR = os.path.join(OUTPUT_DIR, "videos")
PERTURB_DIR = os.path.join(OUTPUT_DIR, "perturbation")
ANALYSIS_DIR = os.path.join(OUTPUT_DIR, "analysis")

SUBGOALS_FILE = os.path.join(ROOT, "subgoals.json")
LABELS_FILE = os.path.join(OUTPUT_DIR, "subgoal_labels.parquet")

# --- the data ---------------------------------------------------------------
# Always pass the revision. Success rates are only comparable against a pinned
# version of the dataset.
DATASET = "lerobot/libero"
DATASET_REVISION = "a1aaacb7f6cd6ee5fb43120f673cebb0cfea7dd4"
SUITE = "libero_10"
N_TASKS = 10

# --- training ---------------------------------------------------------------
POLICY_TYPE = "subgoal_act"
TRAIN_STEPS = 100_000
BATCH_SIZE = 64
SAVE_EVERY = 10_000

# ACT predicts 100 actions at a time, but we only run the first 10 before asking
# again.
#
# This matters more than it looks. ACT's default is to run all 100. ACT was
# tuned on a robot running at 50 Hz, where 100 actions is 2 seconds. LIBERO runs
# at 10 fps, so the same 100 actions is 10 SECONDS, and a 520 step episode would
# contain only about 5 decisions. System 2 could think as often as it liked and
# its answer would almost never reach System 1. At 10 we get about 52 decisions,
# and the closed loop this project is about can actually be seen.
CHUNK_SIZE = 100        # architecture: changing this needs retraining
N_ACTION_STEPS = 10     # inference only: safe to change on a trained policy

# --- evaluation -------------------------------------------------------------
EPISODES_PER_TASK = 10          # 10 tasks x 10 = 100 episodes per seed
EVAL_BATCH_SIZE = 10
SEEDS = [1000, 2000, 3000]      # 3 seeds x 100 = 300 episodes per condition

# --- the ablation conditions ------------------------------------------------
# Only three of these are trained. The rest reuse a trained policy and change
# only where the sub-goal comes from, which is exactly why a swappable
# interface was worth building.
#
#   name          -> (which checkpoint to load, where the sub-goal comes from)
CONDITIONS = {
    # the full system
    "A_learned":  ("A_learned", "learned"),
    # System 2 replaced by perfect knowledge, an upper bound on reasoning
    "A_oracle":   ("A_learned", "oracle"),
    # System 2 replaced by a clock. Cannot go backwards, so it is a clean
    # control for error recovery.
    "A_frozen":   ("A_learned", "frozen"),
    # System 2 replaced by random sub-goals. If this does NOT hurt, System 1 is
    # ignoring us and the whole interface is decoration.
    "A_shuffled": ("A_learned", "shuffled"),
    # the naive baseline named in the exercise: one fixed sub-goal all episode
    "B_static":   ("B_static", "static"),
    # no conditioning at all: the floor, plain ACT
    "C_none":     ("C_none", "none"),
}

# Which conditions need their own training run, and how they are trained.
TRAIN_CONDITIONS = {
    "A_learned": "learned",
    "B_static": "static",
    "C_none": "none",
}

# --- perturbations, for the error-recovery study ----------------------------
PERTURBATIONS = ["none", "forced_drop", "action_noise", "object_shift", "visual_shift"]
PERTURB_AT = 0.4        # fraction of the episode at which to disturb things


def checkpoint_path(condition):
    """Where a condition's trained weights live."""
    checkpoint_name = CONDITIONS[condition][0]
    return os.path.join(CHECKPOINT_DIR, checkpoint_name, "checkpoints", "last", "pretrained_model")


def eval_mode(condition):
    """Where this condition gets its sub-goal from at evaluation time."""
    return CONDITIONS[condition][1]


def needs_simulator_state(condition):
    """True if this condition cannot run through the standard evaluator.

    The oracle reads the simulator's internal state, which the normal policy
    interface never exposes. Only scripts/09_video.py can drive it.
    """
    return eval_mode(condition) == "oracle"
