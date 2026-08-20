"""Shared helpers: the sub-goal index, the progress oracle, and LIBERO glue.

Everything in here is used by more than one script, so it lives in one place.
There are no clever tricks: the tables are small, so plain loops and dicts are
fast enough and much easier to read.
"""

import json
import os

SUITE = "libero_10"

ROOT = os.path.dirname(os.path.abspath(__file__))
SUBGOALS_FILE = os.path.join(ROOT, "subgoals.json")


# ---------------------------------------------------------------------------
# 1. The sub-goal index: two numbers, and how they relate
# ---------------------------------------------------------------------------
# There are two different integers in this project and mixing them up is the
# easiest way to break everything, so they get different names everywhere:
#
#   phase   0, 1, 2, 3      "how far through THIS task am I"
#                           System 2 predicts this. Every metric logs this.
#                           It only makes sense together with the task id.
#
#   skill   0 .. 24         "what am I trying to do right now"
#                           System 1 embeds this. It makes sense on its own.
#                           25 is a special "no sub-goal given" value.
#
# Why two? Because a phase alone is ambiguous across tasks. Phase 2 is "grasp
# the tomato sauce" in task 0 but "place the moka pot" in task 2. ACT gets no
# text input, so if both used the same embedding row it could never tell them
# apart.
#
# But going all the way to one row per (task, phase) is also wrong. That gives
# 35 rows, and then "grasp the white mug" gets TWO different rows, one for
# task 4 and one for task 6, each trained on half as much data. The row would
# also secretly encode which task and which step you are on, which is exactly
# the information a sub-goal is not supposed to carry.
#
# So sub-goals that read the same share a skill id. That gives 25 skills. Task
# 8 ("put both moka pots on the stove") becomes [9, 10, 9, 10]: the first and
# second moka pot are the SAME skill, and System 1 has to look at the picture
# to see which pot is still on the table.
class SubgoalIndex:
    """Looks up which skill a (task, phase) pair means."""

    def __init__(self, data):
        self.tasks = {}          # task id -> list of sub-goal descriptions
        self.skills = {}         # task id -> list of skill ids, same order
        for task_id_text, task in data["tasks"].items():
            task_id = int(task_id_text)
            self.tasks[task_id] = task["subgoals"]
            self.skills[task_id] = task["skills"]

        self.n_skills = data["n_skills"]     # 25
        self.null_skill = self.n_skills      # 25 means "no sub-goal given"
        self.n_rows = self.n_skills + 1      # 26 rows in the embedding table

        # The widest task has 4 sub-goals. This is System 2's output size.
        self.max_phases = max(len(v) for v in self.tasks.values())

        # Instruction text -> task id. LeRobot puts the instruction string in
        # the batch, and it is the only way to recover the task id at eval time.
        self.task_of_instruction = {}
        for task_id, task in data["tasks"].items():
            self.task_of_instruction[task["instruction"].strip()] = int(task_id)

    @classmethod
    def load(cls, path=SUBGOALS_FILE):
        with open(path) as f:
            return cls(json.load(f))

    def n_phases(self, task_id):
        """How many sub-goals this task has. Tasks differ: 2, 3 or 4."""
        return len(self.tasks[int(task_id)])

    def skill(self, task_id, phase):
        """(task, phase) -> skill id. Out-of-range phase means 'no sub-goal'."""
        task_id = int(task_id)
        phase = int(phase)
        if phase < 0 or phase >= self.n_phases(task_id):
            return self.null_skill
        return self.skills[task_id][phase]

    def name(self, task_id, phase):
        """The text of a sub-goal, for printing and for the video overlay."""
        task_id = int(task_id)
        if phase < 0 or phase >= self.n_phases(task_id):
            return "(no sub-goal)"
        return self.tasks[task_id][phase]

    def task_from_instruction(self, instruction):
        """Instruction string -> task id, or None if it is not one of our 10."""
        return self.task_of_instruction.get(str(instruction).strip())


# ---------------------------------------------------------------------------
# 2. LIBERO environment helpers
# ---------------------------------------------------------------------------
# The simulator and the dataset use DIFFERENT names for the same two cameras.
# Normalisation statistics are stored per name, so getting this wrong does not
# crash, it just quietly feeds the policy badly scaled images.
CAMERA_NAMES = {
    "agentview_image": "observation.images.image",              # the scene
    "robot0_eye_in_hand_image": "observation.images.image2",    # the gripper
}


def get_benchmark(suite=SUITE):
    """Load the LIBERO task list. Imported late because it is slow."""
    from libero.libero import benchmark
    return benchmark.get_benchmark_dict()[suite]()


def bddl_path(bench, task_id):
    """Absolute path to a task's .bddl goal definition file."""
    from libero.libero import get_libero_path
    task = bench.get_task(task_id)
    folder = get_libero_path("bddl_files")
    return os.path.join(folder, task.problem_folder, task.bddl_file)


def make_env(bench, task_id, height=256, width=256):
    """Build one LIBERO environment. The caller must close() it."""
    from libero.libero.envs import OffScreenRenderEnv
    env = OffScreenRenderEnv(
        bddl_file_name=bddl_path(bench, task_id),
        camera_heights=height,
        camera_widths=width,
    )
    return env


def reset_to_start(env, bench, task_id, episode=0):
    """Reset to one of LIBERO's stored starting positions.

    Using the stored states (rather than a random reset) is what makes the
    ablation fair: every condition sees the same scenes in the same order, so
    they can be compared episode by episode.
    """
    env.reset()
    states = bench.get_task_init_states(task_id)
    env.set_init_state(states[episode % len(states)])
    # Ten steps, not an arbitrary few. The official LiberoEnv settles the scene
    # with exactly ten no-op frames before handing control to the policy, because
    # right after set_init_state objects can still be floating or intersecting.
    # Matching the count matters: it is part of the starting state, so a different
    # number means a re-rendered episode is not the episode that was measured.
    for _ in range(10):                      # let the scene settle, as lerobot-eval does
        obs, _, _, _ = env.step([0.0] * 6 + [-1.0])   # hand open, do nothing
    return obs


def goal_predicates(env):
    """The task's goal conditions, as written in its .bddl file."""
    return env.env.parsed_problem["goal_state"]


def check_predicates(env, predicates):
    """Which goal conditions are true right now. One True/False per condition.

    This reads the simulator's internal state, which a real robot could not do.
    It is only used to build the oracle condition and to score the others.
    """
    return [bool(env.env._eval_predicate(p)) for p in predicates]


# ---------------------------------------------------------------------------
# 3. The progress oracle
# ---------------------------------------------------------------------------
# For one ablation condition we want to know: if System 2 were PERFECT, how
# well would the robot do? That needs a source of truth for "which phase are we
# actually in", which the oracle provides by reading simulator state.
#
# Two things make this harder than it sounds.
#
# First, the .bddl goal conditions are coarse. They say "the bowl is in the
# drawer", but nothing about "the bowl is currently held". So BDDL alone cannot
# tell grasping apart from placing, which is half of every phase boundary.
# The oracle therefore also looks at the gripper. That is proprioception, not
# privileged knowledge, so the condition stays an honest upper bound.
#
# Second, some conditions are already true at the start. Task 8's stove is on
# at t=0. So the plan below lists ONLY the conditions that mark real progress.
#
# Each task lists:
#   "lead"  conditions that must be true before the pick-and-place part starts
#           (only task 2 has one: turn the stove on first)
#   "place" one condition per object, in the order the objects are handled
ORACLE_PLAN = {
    0: {"lead": [], "place": [0, 1]},
    1: {"lead": [], "place": [0, 1]},
    2: {"lead": [0], "place": [1]},
    3: {"lead": [], "place": [1]},
    4: {"lead": [], "place": [0, 1]},
    5: {"lead": [], "place": [0]},
    6: {"lead": [], "place": [0, 1]},
    7: {"lead": [], "place": [0, 1]},
    8: {"lead": [], "place": [0, 1]},
    9: {"lead": [], "place": [0]},
}

# The gripper opening is state[6] - state[7]. It is cleanly two-valued: about
# 0.006 when closed on something, about 0.080 when open. We use two thresholds
# rather than one so that a value hovering in the middle does not flip back and
# forth every step.
GRIPPER_CLOSED_BELOW = 0.35 * 0.080
GRIPPER_OPEN_ABOVE = 0.65 * 0.080


class SubgoalOracle:
    """Reports the true phase, using simulator state plus the gripper.

    Make ONE of these per episode, because it remembers whether the gripper was
    closed last step.

    Important: n_phases must be THIS TASK's phase count, not the total number
    of skills. Passing 25 here would let it return phases that do not exist,
    which the policy would read as "no sub-goal" and quietly stop conditioning.
    """

    def __init__(self, task_id, n_phases):
        task_id = int(task_id)
        if task_id not in ORACLE_PLAN:
            raise KeyError("no oracle plan for task %d" % task_id)
        self.plan = ORACLE_PLAN[task_id]
        self.n_phases = n_phases
        self.gripper_is_closed = False

    def update_gripper(self, gripper_qpos):
        """Track open/closed with two thresholds, so it does not chatter."""
        opening = gripper_qpos[0] - gripper_qpos[1]
        if self.gripper_is_closed and opening > GRIPPER_OPEN_ABOVE:
            self.gripper_is_closed = False
        elif not self.gripper_is_closed and opening < GRIPPER_CLOSED_BELOW:
            self.gripper_is_closed = True
        return self.gripper_is_closed

    def __call__(self, predicates_true, gripper_qpos):
        """-> the true phase right now.

        The phase counts the same way the training labels do:
        two phases per object (grasp it, then place it).
        """
        closed = self.update_gripper(gripper_qpos)

        # Step 1: any leading sub-goals that are done (e.g. stove turned on).
        phase = 0
        for index in self.plan["lead"]:
            if predicates_true[index]:
                phase += 1
            else:
                # Not done yet, so we are still on this leading sub-goal.
                return min(phase, self.n_phases - 1)

        # Step 2: two phases for every object already placed.
        placed = 0
        for index in self.plan["place"]:
            if predicates_true[index]:
                placed += 1
        phase += 2 * placed

        # Step 3: if we are holding something, the grasp is done and we are
        # part way through the matching place step.
        if closed:
            phase += 1

        return min(phase, self.n_phases - 1)
