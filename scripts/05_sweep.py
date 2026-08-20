"""Step 5: evaluate every condition at every seed.

    python scripts/05_sweep.py
    python scripts/05_sweep.py --dry-run          # just print the plan and the time
    python scripts/05_sweep.py --seeds 1000       # one seed only

This is the ablation study. It takes a few hours, so two things are built in.

It SKIPS anything already finished, so if it stops half way you can just run it
again and it picks up where it left off.

It KEEPS GOING when one run fails, and reports the failures at the end. Dying on
the first problem would throw away hours of results that were already fine.
"""

import argparse
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config                                      # noqa: E402

SECONDS_PER_EPISODE = 10.4          # measured on one machine, for the estimate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--seeds", nargs="+", type=int, default=config.SEEDS)
    args = parser.parse_args()

    # The oracle cannot run here (it needs simulator state), so leave it out and
    # remind the reader at the end.
    runnable = []
    oracle_only = []
    for condition in sorted(config.CONDITIONS):
        if config.needs_simulator_state(condition):
            oracle_only.append(condition)
        else:
            runnable.append(condition)

    # Check the checkpoints exist BEFORE starting, but only skip what is missing
    # rather than refusing to run at all. One untrained condition should not cost
    # us the evaluation of every condition that did train.
    missing = []
    have = []
    for condition in runnable:
        if os.path.isdir(config.checkpoint_path(condition)):
            have.append(condition)
        else:
            missing.append(condition)

    if missing:
        print("SKIPPING, no checkpoint yet: %s" % ", ".join(missing))
    if not have:
        print("nothing to run: no condition has a checkpoint. Train something first.")
        return 1

    total_runs = len(have) * len(args.seeds)
    total_episodes = total_runs * config.N_TASKS * config.EPISODES_PER_TASK
    hours = total_episodes * SECONDS_PER_EPISODE / 3600.0

    print("conditions  %s" % ", ".join(have))
    print("seeds       %s" % ", ".join(str(s) for s in args.seeds))
    print("runs        %d  (%d episodes, roughly %.1f hours)"
          % (total_runs, total_episodes, hours))
    if oracle_only:
        print("not here    %s  (needs scripts/08_video.py)" % ", ".join(oracle_only))

    if args.dry_run:
        return 0

    started = time.time()
    done = 0
    skipped = 0
    failed = []

    for seed in args.seeds:
        for condition in have:
            name = "%s_seed%d" % (condition, seed)
            results_path = os.path.join(config.EVAL_DIR, name, "eval_info.json")

            if os.path.exists(results_path):
                print("[skip] %s already done" % name)
                skipped += 1
                continue

            print("\n[%d/%d] %s" % (done + 1, total_runs, name))
            command = [sys.executable,
                       os.path.join(os.path.dirname(os.path.abspath(__file__)), "04_eval.py"),
                       condition, str(seed)]
            log_path = os.path.join(config.EVAL_DIR, name + ".stdout")
            os.makedirs(config.EVAL_DIR, exist_ok=True)
            with open(log_path, "w") as log:
                code = subprocess.call(command, stdout=log, stderr=subprocess.STDOUT)

            if code == 0 and os.path.exists(results_path):
                with open(results_path) as f:
                    success = json.load(f)["overall"]["pc_success"]
                print("       success %.1f%%" % success)
            else:
                print("       FAILED, see %s" % log_path)
                failed.append(name)
            done += 1

    minutes = (time.time() - started) / 60.0
    print("\nfinished %d runs in %.0f minutes (%d skipped)" % (done, minutes, skipped))
    if failed:
        print("failed: %s" % ", ".join(failed))
        print("run the sweep again to retry only those.")
    if oracle_only:
        print("\nstill to do, the oracle condition:")
        for condition in oracle_only:
            print("  python scripts/08_video.py --condition %s" % condition)
    print("\nNEXT: python scripts/07_analyze.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
