"""Step 2: label every training frame with the phase it belongs to.

    python scripts/02_label_dataset.py
    python scripts/02_label_dataset.py --video 3    # watch task 3 with labels drawn on

System 2 has to be trained to predict the phase, so it needs a label per frame.
LIBERO does not come with those labels, so we work them out from the recorded
demonstrations.

THE SIGNAL IS THE GRIPPER
-------------------------
In pick-and-place, the phase boundaries line up almost exactly with the gripper
opening and closing:

    grasp the bowl          gripper open, closing
    ------ CLOSES ------                             <- boundary
    place it in the drawer  gripper closed, carrying
    ------ OPENS -------                             <- boundary
    close the drawer        gripper open again

So: count the open/close changes, and the phase is how many have happened.

WHY A FIXED THRESHOLD DOES NOT WORK
-----------------------------------
"Closed" is not one number. The fingers stop where the object is, so the closed
reading actually measures how wide the object is. Measured across these tasks it
ranges from 0.006 (a thin bowl rim) to 0.049 (a wide soup can), while "fully
open" is a steady 0.080 everywhere. A single cutoff tuned on one task finds zero
changes on the wide-object tasks, and the labels come out empty with no error.

So the thresholds are worked out per episode, from that episode's own range.

Output: outputs/subgoal_labels.parquet, a separate file joined on the frame index.
We write a separate file rather than adding a column to the dataset, because
rewriting the dataset means regenerating its statistics, and a mistake there
would quietly corrupt training for every condition.
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import SubgoalIndex                                    # noqa: E402
from config import DATASET, DATASET_REVISION, LABELS_FILE, OUTPUT_DIR   # noqa: E402

# "Fully open" is the same everywhere, so it anchors the range.
FULLY_OPEN = 0.080
# If the gripper never closed by at least this much, it never grasped anything.
MIN_MOVEMENT = 0.015
# Two thresholds, as a fraction of the episode's own range, so a reading sitting
# in the middle does not flip back and forth.
CLOSED_FRACTION = 0.35
OPEN_FRACTION = 0.65


def gripper_opening(states):
    """observation.state is [position(3), rotation(3), finger, finger]."""
    return states[:, 6] - states[:, 7]


def find_closed_frames(opening):
    """True wherever the gripper is holding something."""
    low = float(opening.min())
    high = max(float(opening.max()), FULLY_OPEN)
    span = high - low

    if span < MIN_MOVEMENT:
        # The gripper never really closed this episode. Reporting no changes is
        # the honest answer, not a bug.
        return np.zeros(len(opening), dtype=bool)

    closed_below = low + CLOSED_FRACTION * span
    open_above = low + OPEN_FRACTION * span

    closed = np.zeros(len(opening), dtype=bool)
    is_closed = opening[0] < closed_below
    for i in range(len(opening)):
        if is_closed and opening[i] > open_above:
            is_closed = False
        elif not is_closed and opening[i] < closed_below:
            is_closed = True
        closed[i] = is_closed
    return closed


def closed_to_phases(closed, n_phases):
    """Step the phase up at every open/close change.

    We deliberately do NOT force the episode to end on the last phase. If a
    demonstration shows fewer changes than the task has phases, that is a real
    property of the data, and hiding it would hide a wrong split in subgoals.json.
    """
    changes = np.flatnonzero(np.diff(closed.astype(np.int8)) != 0) + 1
    phases = np.zeros(len(closed), dtype=np.int16)
    for change in changes:
        phases[change:] = min(phases[change] + 1, n_phases - 1)
    return phases


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=LABELS_FILE)
    parser.add_argument("--video", type=int, default=None, metavar="TASK",
                        help="also render one episode of this task with labels drawn on")
    args = parser.parse_args()

    index = SubgoalIndex.load()

    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    dataset = LeRobotDataset(DATASET, revision=DATASET_REVISION)
    meta = dataset.meta

    # The dataset holds 40 tasks from all the LIBERO suites. Only 10 are ours,
    # and we find them by matching the instruction text.
    dataset_task_of_instruction = {}
    for row in meta.tasks.index:
        dataset_task_of_instruction[row] = int(meta.tasks.loc[row, "task_index"])

    ours = {}
    for task_id in index.tasks:
        instruction = None
        for text, tid in index.task_of_instruction.items():
            if tid == task_id:
                instruction = text
        if instruction not in dataset_task_of_instruction:
            print("WARNING: instruction not in dataset: %r" % instruction)
            continue
        ours[dataset_task_of_instruction[instruction]] = task_id

    print("matched %d of 10 tasks to the dataset" % len(ours))
    if len(ours) != 10:
        print("FAIL: could not match all 10 tasks")
        return 1

    columns = dataset.hf_dataset.select_columns(
        ["observation.state", "episode_index", "frame_index", "index", "task_index"])

    rows = []
    changes_per_task = {}
    n_episodes = 0

    for episode in range(meta.total_episodes):
        record = meta.episodes[episode]
        start = int(record["dataset_from_index"])
        end = int(record["dataset_to_index"])
        chunk = columns[start:end]

        dataset_task = int(np.asarray(chunk["task_index"])[0])
        if dataset_task not in ours:
            continue                       # not one of our 10 tasks
        task_id = ours[dataset_task]
        n_phases = index.n_phases(task_id)

        states = np.asarray(chunk["observation.state"], dtype=np.float32)
        closed = find_closed_frames(gripper_opening(states))
        phases = closed_to_phases(closed, n_phases)

        found = int(phases.max()) + 1
        changes_per_task.setdefault(task_id, []).append(found)
        n_episodes += 1

        rows.append(pd.DataFrame({
            "index": np.asarray(chunk["index"], dtype=np.int64),
            "episode_index": np.asarray(chunk["episode_index"], dtype=np.int64),
            "frame_index": np.asarray(chunk["frame_index"], dtype=np.int64),
            "task_id": np.int16(task_id),
            "phase": phases,
            "gripper_closed": closed,
        }))

    if not rows:
        print("FAIL: no episodes matched")
        return 1

    table = pd.concat(rows, ignore_index=True)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    table.to_parquet(args.out, index=False)
    print("\nlabelled %d episodes, %d frames -> %s" % (n_episodes, len(table), args.out))

    print("\n%4s %8s %9s  phases found per episode" % ("task", "expected", "episodes"))
    mismatched = 0
    for task_id in sorted(changes_per_task):
        expected = index.n_phases(task_id)
        counts = np.array(changes_per_task[task_id])
        histogram = {}
        for value in np.unique(counts):
            histogram[int(value)] = int((counts == value).sum())
        ok = bool((counts == expected).all())
        mismatched += int((counts != expected).sum())
        print("%4d %8d %9d  %s  %s"
              % (task_id, expected, len(counts), histogram, "ok" if ok else "MISMATCH"))

    print("\n%d of %d episodes (%.1f%%) found a different number of phases than expected."
          % (mismatched, n_episodes, 100.0 * mismatched / n_episodes))
    print("Some mismatch is normal: a re-grasp adds an extra change, and 'turn on the")
    print("stove' or 'close the drawer' need no grasp at all. A HIGH rate on one task")
    print("means that task's split in subgoals.json is probably wrong.")
    print("\nNEXT: look at it before training on it:")
    print("      python scripts/02_label_dataset.py --video 3")

    if args.video is not None:
        return render_video(dataset, table, index, args.video)
    return 0


def render_video(dataset, table, index, task_id):
    """Draw the assigned phase on a real episode and save it as an MP4."""
    import cv2

    task_rows = table[table.task_id == task_id]
    if len(task_rows) == 0:
        print("no labelled frames for task %d" % task_id)
        return 1

    episode = int(task_rows.episode_index.iloc[0])
    episode_rows = task_rows[task_rows.episode_index == episode].sort_values("frame_index")
    names = index.tasks[task_id]

    folder = os.path.join(OUTPUT_DIR, "labels")
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, "labels_task%d_ep%d.mp4" % (task_id, episode))
    writer = None

    print("\nrendering episode %d of task %d (%d frames)..." % (episode, task_id, len(episode_rows)))
    for _, row in episode_rows.iterrows():
        sample = dataset[int(row["index"])]
        image = sample["observation.images.image"].permute(1, 2, 0).numpy() * 255
        image = np.ascontiguousarray(image.astype(np.uint8))

        current = int(row["phase"])
        for i, name in enumerate(names):
            colour = (80, 255, 80) if i == current else (150, 150, 150)
            marker = ">" if i == current else " "
            text = ("%s %d. %s" % (marker, i, name))[:44]
            position = (6, 16 + 15 * i)
            cv2.putText(image, text, position, cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 0, 0), 3)
            cv2.putText(image, text, position, cv2.FONT_HERSHEY_SIMPLEX, 0.38, colour, 1)

        grip = "CLOSED" if row["gripper_closed"] else "open"
        cv2.putText(image, "gripper %s" % grip, (6, image.shape[0] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1)

        if writer is None:
            size = (image.shape[1], image.shape[0])
            writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 20, size)
        writer.write(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

    if writer is not None:
        writer.release()
    print("wrote %s" % path)
    print("WATCH IT. The highlighted line must match what the arm is doing.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
