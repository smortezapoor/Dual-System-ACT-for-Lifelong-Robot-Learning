"""Step 4: evaluate one condition at one seed.

    python scripts/04_eval.py A_learned 1000
    python scripts/04_eval.py A_learned 1000 --quick     # 2 episodes per task

This runs 10 episodes on each of the 10 tasks, so 100 episodes per seed. With
three seeds that is 300 episodes per condition, which is what the reported
numbers are based on.

THE COMPARISON IS PAIRED
------------------------
We pass --env.init_states=true and a fixed seed, so every condition sees the
same starting scenes in the same order. That means we can compare them episode
by episode instead of comparing two averages, which is far more sensitive: the
variation caused by some scenes being harder than others cancels out.
"""

import argparse
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config                                      # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("condition", choices=sorted(config.CONDITIONS))
    parser.add_argument("seed", type=int, nargs="?", default=1000)
    parser.add_argument("--quick", action="store_true", help="2 episodes per task")
    args = parser.parse_args()

    episodes = config.EPISODES_PER_TASK
    batch_size = config.EVAL_BATCH_SIZE
    if args.quick:
        episodes = 2
        batch_size = 2
        print("QUICK MODE: 2 episodes per task. Not for reporting.")

    # A batch size of 1 ALWAYS reports 0.0% success, whatever the policy does.
    #
    # Measured on one trained policy over tasks 3, 4 and 9:
    #     batch size 1  ->   0.0%
    #     batch size 2  ->  50.0%
    #     batch size 3  ->  66.7%
    #
    # Something in the evaluator's success check drops the flag when there is
    # only one environment. It fails quietly: the rollout runs, the videos look
    # fine, and every condition scores exactly zero, which reads as "the policy
    # learned nothing" rather than "the harness is misconfigured". Refuse,
    # because a believable wrong number costs far more than a crash.
    if batch_size < 2:
        print("batch size %d would report 0.0%% regardless of the policy. Use 2 or more."
              % batch_size)
        return 1

    # The oracle needs to read the simulator's internal state, which the normal
    # evaluator never exposes to the policy. Say so now, not three hours into a
    # sweep.
    if config.needs_simulator_state(args.condition):
        print("'%s' cannot run through the standard evaluator: it needs the true" % args.condition)
        print("progress from the simulator, which the policy interface does not provide.")
        print("Run it through the rollout script instead:")
        print("  python scripts/08_video.py --condition %s" % args.condition)
        return 1

    checkpoint = config.checkpoint_path(args.condition)
    if not os.path.isdir(checkpoint):
        print("no checkpoint at %s" % checkpoint)
        print("train it first: python scripts/03_train.py %s"
              % config.CONDITIONS[args.condition][0])
        return 1

    mode = config.eval_mode(args.condition)
    output_dir = os.path.join(config.EVAL_DIR, "%s_seed%d" % (args.condition, args.seed))

    print("condition   %s" % args.condition)
    print("checkpoint  %s" % checkpoint)
    print("sub-goal    %s" % mode)
    print("episodes    %d per task x %d tasks" % (episodes, config.N_TASKS))
    print("output      %s" % output_dir)

    command = [
        "lerobot-eval",
        "--policy.path=%s" % checkpoint,
        "--env.type=libero",
        "--env.task=%s" % config.SUITE,
        "--env.observation_height=256",
        "--env.observation_width=256",
        "--env.init_states=true",              # same scenes for every condition
        "--env.control_mode=relative",
        "--env.max_parallel_tasks=1",
        "--eval.batch_size=%d" % batch_size,
        "--eval.n_episodes=%d" % episodes,
        "--seed=%d" % args.seed,
        "--output_dir=%s" % output_dir,
        # Swap where the sub-goal comes from. This is the whole reason six
        # conditions come out of three training runs.
        "--policy.eval_mode=%s" % mode,
        "--policy.subgoals_path=%s" % config.SUBGOALS_FILE,
        # Override how much of each predicted chunk we actually run. See the
        # note in config.py: the trained default is far too long for 10 fps.
        "--policy.n_action_steps=%d" % config.N_ACTION_STEPS,
    ]

    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, "eval.log")

    with open(log_path, "w") as log:
        process = subprocess.Popen(command, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True)
        for line in process.stdout:
            sys.stdout.write(line)
            log.write(line)
        process.wait()

    results_path = os.path.join(output_dir, "eval_info.json")
    if not os.path.exists(results_path):
        print("\nFAILED: no eval_info.json written. See %s" % log_path)
        return 1

    with open(results_path) as f:
        results = json.load(f)
    print("\nsuccess: %.1f%%" % results["overall"]["pc_success"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
