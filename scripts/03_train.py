"""Step 3: train one condition.

    python scripts/03_train.py A_learned
    python scripts/03_train.py A_learned --smoke     # 10 steps, to check the setup

Only three conditions are trained. The other three reuse one of these and change
only where the sub-goal comes from at evaluation time, which is the whole point
of keeping the interface a single swappable number.

    A_learned   System 2 predicts the sub-goal, trained alongside System 1
    B_static    always the task's first sub-goal (the naive baseline)
    C_none      no sub-goal at all (plain ACT, the floor)

WHY B_static AND C_none GET THEIR OWN RUNS
------------------------------------------
It would be cheaper to train A_learned once and fake the other two at evaluation
time. That would be wrong, and in a way that flatters the full system.

A policy trained on changing sub-goals has never seen a constant one, so feeding
it a constant is unlike anything in its training data, and it would do badly for
that reason rather than because a constant sub-goal is a bad idea. Same for
feeding "no sub-goal" to a policy that always had one. Each baseline has to be
trained on its own terms to be a fair comparison.

This calls lerobot-train, which does the actual training. We only assemble the
arguments.
"""

import argparse
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config                                        # noqa: E402
from common import SubgoalIndex                      # noqa: E402


def get_libero10_episodes():
    """Which dataset episodes belong to libero_10.

    WHY THIS MATTERS. lerobot/libero holds 1,693 episodes across all 40 LIBERO
    tasks, but we only ever evaluate on 10 of them. Training on all 1,693 makes
    every condition worse, and it hurts the unconditioned ones most: a policy
    with no text input already cannot tell which task it is doing, and adding 30
    unrelated behaviours makes that strictly harder.

    That is not hypothetical. The first baseline trained on all 1,693 episodes
    and scored 1 out of 100. A floor has to be a fair floor.
    """
    cache = os.path.join(config.OUTPUT_DIR, "libero10_episodes.json")
    if os.path.exists(cache):
        with open(cache) as f:
            return json.load(f)["episodes"]

    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    index = SubgoalIndex.load()
    dataset = LeRobotDataset(config.DATASET, revision=config.DATASET_REVISION)
    meta = dataset.meta

    our_dataset_tasks = set()
    for row in meta.tasks.index:
        if index.task_from_instruction(row) is not None:
            our_dataset_tasks.add(int(meta.tasks.loc[row, "task_index"]))

    episodes = []
    for episode in range(meta.total_episodes):
        record = meta.episodes[episode]
        start = int(record["dataset_from_index"])
        task_index = int(dataset.hf_dataset[start]["task_index"])
        if task_index in our_dataset_tasks:
            episodes.append(episode)

    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    with open(cache, "w") as f:
        json.dump({"n_episodes": len(episodes), "episodes": episodes}, f)
    return episodes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("condition", choices=sorted(config.TRAIN_CONDITIONS))
    parser.add_argument("--smoke", action="store_true",
                        help="10 steps at batch size 2, just to prove the config works")
    args = parser.parse_args()

    train_mode = config.TRAIN_CONDITIONS[args.condition]
    steps = config.TRAIN_STEPS
    batch_size = config.BATCH_SIZE
    name = args.condition

    if args.smoke:
        steps = 10
        batch_size = 2
        name = args.condition + "_smoke"     # so it cannot overwrite a real run
        print("SMOKE TEST: 10 steps, to check the configuration before a long run")

    if not os.path.exists(config.SUBGOALS_FILE):
        print("MISSING %s -- run scripts/01_make_subgoals.py" % config.SUBGOALS_FILE)
        return 1
    if not os.path.exists(config.LABELS_FILE):
        print("MISSING %s -- run scripts/02_label_dataset.py" % config.LABELS_FILE)
        return 1

    index = SubgoalIndex.load()
    episodes = get_libero10_episodes()
    output_dir = os.path.join(config.CHECKPOINT_DIR, name)

    # lerobot-train refuses to write into a folder that already exists, so say
    # so clearly rather than letting it fail with a longer message.
    if os.path.exists(output_dir):
        print("output folder already exists: %s" % output_dir)
        print("delete it, or pass --resume=true to lerobot-train yourself.")
        return 1

    print("condition   %s (train_mode=%s)" % (args.condition, train_mode))
    print("skills      %d (+1 for 'no sub-goal')" % index.n_skills)
    print("max phases  %d" % index.max_phases)
    print("episodes    %d (libero_10 only, out of 1693 in the dataset)" % len(episodes))
    print("steps       %d at batch size %d" % (steps, batch_size))
    print("output      %s" % output_dir)

    command = [
        "lerobot-train",
        "--policy.type=%s" % config.POLICY_TYPE,
        "--policy.push_to_hub=false",
        "--policy.device=cuda",
        "--policy.train_mode=%s" % train_mode,
        "--policy.eval_mode=%s" % train_mode,
        "--policy.n_skills=%d" % index.n_skills,
        "--policy.max_phases=%d" % index.max_phases,
        "--policy.subgoals_path=%s" % config.SUBGOALS_FILE,
        "--policy.labels_path=%s" % config.LABELS_FILE,
        "--policy.chunk_size=%d" % config.CHUNK_SIZE,
        "--dataset.repo_id=%s" % config.DATASET,
        "--dataset.revision=%s" % config.DATASET_REVISION,
        "--dataset.episodes=%s" % json.dumps(episodes),
        "--steps=%d" % steps,
        "--batch_size=%d" % batch_size,
        "--save_freq=%d" % config.SAVE_EVERY,
        "--output_dir=%s" % output_dir,
    ]

    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    log_path = output_dir + ".train.log"

    print("\nrunning: %s\n" % " ".join(command))
    with open(log_path, "w") as log:
        process = subprocess.Popen(command, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True)
        for line in process.stdout:
            sys.stdout.write(line)
            log.write(line)
        code = process.wait()

    if code != 0:
        print("\nFAILED (exit %d). See %s" % (code, log_path))
        return code

    saved = os.path.join(output_dir, "checkpoints", "last", "pretrained_model")
    print("\ndone: %s" % saved)
    print("check the loss went down:  grep -o 'loss:[0-9.]*' %s | tail -5" % log_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
