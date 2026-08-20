"""Looks up which skill a (task, phase) pair means.

I keep this as a deliberate near-copy of the class in common.py, and the
reason for duplicating it rather than importing one from the other is worth
stating.

This package has to install and run on its own, without the rest of the repo
(that is what makes it a proper LeRobot plugin instead of a fork). The scripts
in the repo have to run without this package installed. So neither side can
import the other, and a small amount of duplication is the price.

It is only data lookup with no dependencies, and scripts/00_check_setup.py
checks that the two copies agree on every task, so a silent drift is caught.
"""

import json


class SubgoalIndex:
    """Two integers, and how they relate.

        phase   0..3     how far through THIS task the episode is. System 2
                         predicts it, and every metric logs it. Only meaningful
                         together with the task id.

        skill   0..24    what the robot is trying to do. System 1 embeds it. It
                         stands on its own, and 25 means "no sub-goal given".

    Sub-goals that read the same share a skill id, so "grasp the white mug" is
    one embedding row whether it appears in task 4 or task 6, and it trains on
    both tasks' data.
    """

    def __init__(self, data):
        self.tasks = {}        # task id -> list of sub-goal descriptions
        self.skills = {}       # task id -> list of skill ids, in the same order
        for task_id_text, task in data["tasks"].items():
            task_id = int(task_id_text)
            self.tasks[task_id] = task["subgoals"]
            self.skills[task_id] = task["skills"]

        self.n_skills = data["n_skills"]      # 25
        self.null_skill = self.n_skills       # 25 means "no sub-goal"
        self.max_phases = max(len(v) for v in self.tasks.values())

        # LeRobot puts the instruction text in the batch, and that is the only
        # way to tell which task is running at evaluation time.
        self.task_of_instruction = {}
        for task_id_text, task in data["tasks"].items():
            self.task_of_instruction[task["instruction"].strip()] = int(task_id_text)

    @classmethod
    def load(cls, path):
        with open(path) as f:
            return cls(json.load(f))

    def n_phases(self, task_id):
        """How many sub-goals this task has. Tasks have 2, 3 or 4."""
        return len(self.tasks[int(task_id)])

    def skill(self, task_id, phase):
        """(task, phase) -> skill id. A phase that does not exist means 'none'."""
        task_id = int(task_id)
        phase = int(phase)
        if phase < 0 or phase >= self.n_phases(task_id):
            return self.null_skill
        return self.skills[task_id][phase]

    def name(self, task_id, phase):
        """The sub-goal text, for printing and for the video overlay."""
        task_id = int(task_id)
        if phase < 0 or phase >= self.n_phases(task_id):
            return "(no sub-goal)"
        return self.tasks[task_id][phase]

    def task_from_instruction(self, instruction):
        """Instruction text -> task id, or None if it is not one of the 10."""
        return self.task_of_instruction.get(str(instruction).strip())
