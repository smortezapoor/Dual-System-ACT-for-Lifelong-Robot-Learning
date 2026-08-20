"""Step 1: split each task instruction into ordered sub-goals, and assign skills.

    python scripts/01_make_subgoals.py            # write subgoals.json
    python scripts/01_make_subgoals.py --check    # check the committed file

This runs ONCE and its output is committed, because it does not depend on any
training. Everything after this reads subgoals.json.

WHY RULES AND NOT A LANGUAGE MODEL
----------------------------------
The LIBERO-Long instructions are already built out of parts: "put both X and Y
in the basket", "turn on the stove and put the moka pot on it". Splitting those
is the part a rule can do almost perfectly. Asking a language model instead
would add a dependency, add a way to fail, and need checking by hand anyway,
across only ten tasks that fit on one screen.

There are three things that make naive splitting on " and " wrong, and all
three actually occur in these ten tasks:

  1. Some object NAMES contain "and": "the yellow and white mug" is one object.
     We detect this by checking the words against the object names in the task's
     .bddl goal file, rather than keeping a list of special cases.
  2. Some instructions use "it": "...and close it", "put the moka pot on it".
  3. Some are plural: "put both moka pots on the stove" means two objects, and
     the .bddl file confirms it by listing moka_pot_1 and moka_pot_2.

Read the output. Ten tasks is small enough to check by eye, and editing
subgoals.json by hand is expected and supported.
"""

import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import bddl_path, get_benchmark          # noqa: E402
from config import SUBGOALS_FILE                     # noqa: E402

ORDINALS = ["first", "second", "third", "fourth"]

# Words that appear in object names but carry no meaning for matching.
NAME_NOISE = {"the", "a", "an", "1", "2", "of", "region", "contain", "cook", "heating"}


# ---------------------------------------------------------------------------
# Reading the object names out of the task definition
# ---------------------------------------------------------------------------
def get_object_names(bench, task_id):
    """The object names in this task's goal, e.g. ['white_yellow_mug_1', ...]."""
    with open(bddl_path(bench, task_id)) as f:
        text = f.read()
    # Find the (:goal ...) block, then pull out every name ending in a number.
    match = re.search(r"\(:goal(.*?)\n\s*\)\s*\)?\s*$", text, re.S)
    goal_text = match.group(1) if match else text
    return re.findall(r"[A-Za-z_][A-Za-z0-9_]*_\d+", goal_text)


def name_words(object_name):
    """'white_yellow_mug_1' -> {'white', 'yellow', 'mug'}."""
    words = set()
    for word in object_name.lower().split("_"):
        if word and not word.isdigit() and word not in NAME_NOISE:
            words.add(word)
    return words


# ---------------------------------------------------------------------------
# Splitting an instruction into clauses
# ---------------------------------------------------------------------------
def hide_ands_inside_names(text, object_names):
    """Protect the 'and' in a name like 'the yellow and white mug'.

    We look at every "WORD and WORD WORD" in the instruction and ask: do those
    three words all belong to one real object? If they do, that "and" is part
    of a name, so we swap it for a placeholder that survives the clause split
    and gets swapped back at the end.
    """
    all_object_words = [name_words(name) for name in object_names]

    # Work right to left, so replacing text does not shift the positions of the
    # matches we have not handled yet.
    matches = list(re.finditer(r"(\w+) and (\w+) (\w+)", text))
    for match in reversed(matches):
        three_words = {match.group(1).lower(), match.group(2).lower(), match.group(3).lower()}
        belongs_to_one_object = False
        for object_words in all_object_words:
            if three_words <= object_words:
                belongs_to_one_object = True
        if belongs_to_one_object:
            safe = "%s <AND> %s %s" % (match.group(1), match.group(2), match.group(3))
            text = text[:match.start()] + safe + text[match.end():]
    return text


def split_into_clauses(instruction, object_names):
    """One instruction -> a list of simple clauses."""
    text = instruction.strip().rstrip(".")
    text = hide_ands_inside_names(text, object_names)

    # "put both A and B in the basket" -> two clauses, one per object.
    match = re.match(r"^put both (?:the )?(.+?) and (?:the )?(.+?) (in|on) (?:the )?(.+)$", text)
    if match:
        first, second, preposition, destination = match.groups()
        return [
            "put the %s %s the %s" % (first, preposition, destination),
            "put the %s %s the %s" % (second, preposition, destination),
        ]

    # "put both moka pots on the stove" -> one clause per matching object.
    match = re.match(r"^put both (?:the )?(.+?)s (in|on) (?:the )?(.+)$", text)
    if match:
        noun, preposition, destination = match.groups()
        last_word = noun.split()[-1]
        count = 0
        for name in object_names:
            if last_word in name.lower():
                count += 1
        if count == 0:
            count = 2
        clauses = []
        for i in range(count):
            clauses.append("put the %s %s %s the %s" % (ORDINALS[i], noun, preposition, destination))
        return clauses

    # Otherwise, every remaining " and " really is a clause boundary.
    return [part.strip() for part in re.split(r"\s+and\s+", text)]


# ---------------------------------------------------------------------------
# Turning clauses into sub-goals
# ---------------------------------------------------------------------------
PUT_PATTERN = re.compile(
    r"^put (?:the )?(.+?) (in|on|to the right of|to the left of) (?:the )?(.+)$")
PICK_PATTERN = re.compile(r"^pick up (?:the )?(.+?)(?: and place it (in|on) (?:the )?(.+))?$")
TURN_ON_PATTERN = re.compile(r"^turn on (?:the )?(.+)$")
CLOSE_PATTERN = re.compile(r"^close (?:the )?(.+)$")


def replace_pronouns(clauses):
    """Turn 'close it' into 'close the microwave', and so on."""
    result = []
    last_object = None
    last_destination = None

    for clause in clauses:
        # "close it" / "close them"
        if re.fullmatch(r"close (it|them)", clause):
            target = last_destination or last_object
            if target:
                clause = "close the %s" % target
        # "put the moka pot on it", "place it in the caddy"
        elif " it" in clause:
            target = last_destination or last_object
            if target:
                clause = re.sub(r"\b(on|in) it\b", r"\1 the " + target, clause)
            if last_object:
                clause = re.sub(r"\bplace it\b", "place the " + last_object, clause)

        # Remember what this clause was about, for the next clause's pronouns.
        match = PUT_PATTERN.match(clause) or PICK_PATTERN.match(clause)
        if match:
            last_object = match.group(1).strip()
            if match.lastindex and match.lastindex >= 3 and match.group(3):
                last_destination = match.group(3).strip()
        else:
            match = TURN_ON_PATTERN.match(clause)
            if match:
                last_object = match.group(1).strip()
                last_destination = last_object

        result.append(clause)
    return result


def clause_to_subgoals(clause):
    """One clause -> one or two sub-goals.

    Moving an object is always TWO sub-goals, grasp then place, because that is
    where the gripper opens and closes, and that is what the training labels can
    actually detect. Turning something on or closing a door is one.
    """
    clause = clause.strip().rstrip(".")

    match = PUT_PATTERN.match(clause)
    if match:
        obj, preposition, destination = [g.strip() for g in match.groups()]
        return ["grasp the %s" % obj,
                "place the %s %s the %s" % (obj, preposition, destination)]

    match = PICK_PATTERN.match(clause)
    if match:
        obj = match.group(1).strip()
        preposition = match.group(2)
        destination = match.group(3)
        if destination:
            return ["grasp the %s" % obj,
                    "place the %s %s the %s" % (obj, preposition, destination.strip())]
        return ["grasp the %s" % obj]

    match = TURN_ON_PATTERN.match(clause)
    if match:
        return ["turn on the %s" % match.group(1).strip()]

    match = CLOSE_PATTERN.match(clause)
    if match:
        return ["close the %s" % match.group(1).strip()]

    # Not recognised. Keep it as-is so a human notices when reading the output.
    return [clause]


def decompose(instruction, object_names):
    """One instruction -> its ordered list of sub-goals."""
    clauses = split_into_clauses(instruction, object_names)
    clauses = replace_pronouns(clauses)

    subgoals = []
    for clause in clauses:
        for subgoal in clause_to_subgoals(clause):
            subgoal = subgoal.replace("<AND>", "and")
            if subgoal not in subgoals:
                subgoals.append(subgoal)
    return subgoals


# ---------------------------------------------------------------------------
# Assigning skill ids
# ---------------------------------------------------------------------------
# Two sub-goals get the same skill id when they describe the same action on the
# same object. We compare a cleaned-up version of the text: lower case, articles
# removed, and ordinal words removed.
#
# Removing ordinals is the interesting choice. It makes "grasp the first moka
# pot" and "grasp the second moka pot" the SAME skill. "First" and "second" only
# say which repetition we are on, which is exactly the progress information the
# sub-goal is not supposed to carry: System 1 should look at the picture to see
# which pot is still on the table.
#
# Note what this does NOT merge. "place the white mug on the left plate" and
# "place the white mug on the plate" have different destinations, so they stay
# separate. Comparing cleaned text keeps them apart for free. A similarity score
# would have merged them, because they are 90% the same words.
SKILL_NOISE = {"the", "a", "an", "of"}
ORDINAL_PATTERN = re.compile(r"\b(first|second|third|fourth)\b")


def skill_key(subgoal_text):
    """The cleaned text used to decide whether two sub-goals are the same skill."""
    text = subgoal_text.lower()
    text = ORDINAL_PATTERN.sub(" ", text)
    words = []
    for word in re.findall(r"[a-z]+", text):
        if word not in SKILL_NOISE:
            words.append(word)
    return " ".join(words)


def assign_skills(tasks):
    """Give every sub-goal a skill id, numbering them 0, 1, 2, ... in order."""
    skill_id_of_key = {}
    for task_id in sorted(tasks, key=int):
        skill_ids = []
        for subgoal in tasks[task_id]["subgoals"]:
            key = skill_key(subgoal)
            if key not in skill_id_of_key:
                skill_id_of_key[key] = len(skill_id_of_key)
            skill_ids.append(skill_id_of_key[key])
        tasks[task_id]["skills"] = skill_ids
    return len(skill_id_of_key)


# ---------------------------------------------------------------------------
def build():
    bench = get_benchmark()
    tasks = {}
    for task_id in range(bench.n_tasks):
        task = bench.get_task(task_id)
        object_names = get_object_names(bench, task_id)
        tasks[str(task_id)] = {
            "instruction": task.language,
            "subgoals": decompose(task.language, object_names),
        }
    n_skills = assign_skills(tasks)
    return {
        "suite": "libero_10",
        "note": "Hand-audited. Edit directly if a split is wrong.",
        "n_skills": n_skills,
        "tasks": tasks,
    }


def report(data):
    """Print everything, and flag anything that looks wrong."""
    problems = []
    for task_id in sorted(data["tasks"], key=int):
        task = data["tasks"][task_id]
        print("\n[%s] %s" % (task_id, task["instruction"]))
        for i, subgoal in enumerate(task["subgoals"]):
            skill = task["skills"][i]
            print("    phase %d  skill %2d  %s" % (i, skill, subgoal))

            # An unresolved pronoun means replace_pronouns missed one.
            if re.search(r"\b(it|them)\b", subgoal):
                problems.append("task %s: unresolved pronoun in %r" % (task_id, subgoal))
            # Everything should start with one of our four action words.
            if not re.match(r"^(grasp|place|close|turn on)\b", subgoal):
                problems.append("task %s: unexpected wording in %r" % (task_id, subgoal))

    # Show which sub-goals ended up sharing a skill. This is the interesting bit.
    print("\nskills shared by more than one sub-goal:")
    for skill in range(data["n_skills"]):
        members = []
        for task_id in sorted(data["tasks"], key=int):
            task = data["tasks"][task_id]
            for phase, skill_id in enumerate(task["skills"]):
                if skill_id == skill:
                    members.append("task %s phase %d" % (task_id, phase))
        if len(members) > 1:
            name = None
            for task_id in sorted(data["tasks"], key=int):
                task = data["tasks"][task_id]
                if skill in task["skills"]:
                    name = task["subgoals"][task["skills"].index(skill)]
                    break
            print("   skill %2d  %-38s %s" % (skill, name, ", ".join(members)))
    return problems


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=SUBGOALS_FILE)
    parser.add_argument("--check", action="store_true",
                        help="check the committed file instead of rebuilding it")
    args = parser.parse_args()

    if args.check:
        if not os.path.exists(args.out):
            print("MISSING %s" % args.out)
            return 1
        with open(args.out) as f:
            data = json.load(f)
    else:
        data = build()

    problems = report(data)

    total = 0
    for task in data["tasks"].values():
        total += len(task["subgoals"])
    print("\n%d tasks, %d sub-goals, %d distinct skills"
          % (len(data["tasks"]), total, data["n_skills"]))

    if len(data["tasks"]) != 10:
        print("FAIL: expected 10 tasks")
        return 1

    # Skill ids must be 0, 1, 2, ... with no gaps. A gap would mean an embedding
    # row that never gets used and a table sized wrongly.
    used = set()
    for task in data["tasks"].values():
        used.update(task["skills"])
    if used != set(range(data["n_skills"])):
        print("FAIL: skill ids have gaps")
        return 1

    if problems:
        print("\nCHECK THESE BY HAND:")
        for problem in problems:
            print("  - %s" % problem)
        return 1

    if not args.check:
        with open(args.out, "w") as f:
            json.dump(data, f, indent=2)
        print("\nwrote %s" % args.out)
    print("\nOK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
