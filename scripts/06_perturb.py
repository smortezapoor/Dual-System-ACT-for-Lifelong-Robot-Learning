"""Step 6: the error-recovery study.

    python scripts/06_perturb.py --condition A_learned
    python scripts/06_perturb.py --condition A_learned --perturbation forced_drop
    python scripts/06_perturb.py --list

This answers the second question in the exercise: does System 2 actually help
System 1 recover when something goes wrong?

We break something on purpose part way through an episode, then measure two
different things:

    did it recover?          success rate, compared with an undisturbed run
    did it NOTICE?           how often System 2 moved its phase backwards

Those two are worth separating. If System 2 goes backwards after a drop, it
detected the problem. Whether the robot then succeeds is a separate question
about System 1. Measuring only success would blur them together, and the whole
point of the exercise is to tell reasoning failures from execution failures.

A_frozen is the control here: it follows a clock, so it CANNOT go backwards. Any
difference in backwards counts between it and A_learned is caused by the
mechanism, not by chance.

Output: outputs/perturbation/<condition>/<perturbation>.jsonl, one line per episode.
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config                                                   # noqa: E402
from common import (SubgoalIndex, SubgoalOracle, check_predicates,   # noqa: E402
                    get_benchmark, goal_predicates, make_env, reset_to_start)
from rollout import Runner, count_transitions                   # noqa: E402

MAX_STEPS = 520

DESCRIPTIONS = {
    "none": "nothing goes wrong (the control)",
    "forced_drop": "force the gripper open while carrying (the main case)",
    "action_noise": "add random noise to the actions for 20 steps",
    "object_shift": "teleport the target object a few centimetres",
    "visual_shift": "make the camera image darker",
}


# ---------------------------------------------------------------------------
# The four ways of breaking things
# ---------------------------------------------------------------------------
def force_drop(action):
    """Open the gripper, whatever the policy asked for."""
    changed = action.copy()
    changed[6] = -1.0            # channel 6 is the gripper; -1 is fully open
    return changed


def add_noise(action, random_state, scale=0.5):
    """Add random noise, then keep the action in its valid range."""
    noisy = action + random_state.normal(0, scale, size=action.shape)
    return np.clip(noisy, -1.0, 1.0)


def shift_object(env, random_state, distance=0.05):
    """Move the target object a few centimetres. True if something moved."""
    try:
        sim = env.env.sim
        predicates = env.env.parsed_problem["goal_state"]

        # Find the first real object mentioned in the goal, skipping the names
        # that describe places rather than things.
        target = None
        for predicate in predicates:
            for argument in predicate[1:]:
                if "region" not in argument and target is None:
                    target = argument
        if target is None:
            return False

        for body_name in sim.model.body_names:
            if target in body_name:
                body_id = sim.model.body_name2id(body_name)
                joint_id = sim.model.body_jntadr[body_id]
                if joint_id < 0:
                    return False          # bolted down, cannot be moved
                address = sim.model.jnt_qposadr[joint_id]
                sim.data.qpos[address:address + 2] += random_state.normal(0, distance, size=2)
                sim.forward()             # recompute the physics after our edit
                return True
    except Exception as error:
        print("  object_shift failed: %s" % error)
    return False


def darken_images(obs, factor=0.55):
    """Make the camera images darker, without touching anything else."""
    changed = dict(obs)
    for key in ("agentview_image", "robot0_eye_in_hand_image"):
        if key in changed:
            darker = changed[key].astype(np.float32) * factor
            changed[key] = np.clip(darker, 0, 255).astype(np.uint8)
    return changed


# ---------------------------------------------------------------------------
def run_episode(runner, index, bench, task_id, episode, perturbation, random_state,
                perturb_at=0.4):
    """Run one episode, breaking something part way through."""
    env = make_env(bench, task_id)
    try:
        obs = reset_to_start(env, bench, task_id, episode)
        runner.reset()

        instruction = bench.get_task(task_id).language
        predicates = goal_predicates(env)
        n_phases = index.n_phases(task_id)

        # The oracle needs its own instance per episode, and it must be told
        # THIS TASK's phase count. Passing the total number of skills would let
        # it return phases that do not exist, which the policy reads as "no
        # sub-goal" and quietly stops conditioning for most of the episode.
        oracle = SubgoalOracle(task_id, n_phases)
        uses_oracle = runner.config.eval_mode == "oracle"

        perturb_step = int(MAX_STEPS * perturb_at)
        noise_until = perturb_step + 20
        injected = False

        phases = []
        success = False
        steps_run = MAX_STEPS

        for step in range(MAX_STEPS):
            truth = check_predicates(env, predicates)
            true_phase = oracle(truth, obs["robot0_gripper_qpos"])
            if uses_oracle:
                runner.set_oracle_phase(true_phase)

            view = obs
            if perturbation == "visual_shift" and step >= perturb_step:
                view = darken_images(obs)

            action = runner.act(view, instruction)

            choice = runner.last_choice()
            if choice is not None:
                phases.append(choice["phase"])

            # Break something, once, at the chosen moment.
            if step == perturb_step and not injected:
                if perturbation == "object_shift":
                    injected = shift_object(env, random_state)
                else:
                    injected = perturbation != "none"

            if perturbation == "forced_drop" and perturb_step <= step < perturb_step + 10:
                action = force_drop(action)
            if perturbation == "action_noise" and perturb_step <= step < noise_until:
                action = add_noise(action, random_state)

            obs, _, done, _ = env.step(action.tolist())

            if all(check_predicates(env, predicates)):
                success = True
                steps_run = step + 1
                break
            if done:
                steps_run = step + 1
                break

        forwards, backwards = count_transitions(phases)
        return {
            "task_id": task_id,
            "episode": episode,
            "perturbation": perturbation,
            "success": success,
            "steps_run": steps_run,
            "phases": phases,
            "n_forward": forwards,
            "n_backward": backwards,
            "phase_at_end": phases[-1] if phases else None,
        }
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", default="A_learned", choices=sorted(config.CONDITIONS))
    parser.add_argument("--perturbation", default=None, choices=config.PERTURBATIONS)
    parser.add_argument("--tasks", default=None, help="comma-separated task ids")
    parser.add_argument("--episodes", type=int, default=5, help="episodes per task")
    parser.add_argument("--perturb-at", type=float, default=config.PERTURB_AT)
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()

    if args.list:
        print("perturbations:")
        for name in config.PERTURBATIONS:
            print("  %-14s %s" % (name, DESCRIPTIONS[name]))
        return 0

    checkpoint = config.checkpoint_path(args.condition)
    if not os.path.isdir(checkpoint):
        print("no checkpoint at %s" % checkpoint)
        return 1

    if args.tasks:
        tasks = [int(t) for t in args.tasks.split(",")]
    else:
        tasks = list(range(config.N_TASKS))

    perturbations = [args.perturbation] if args.perturbation else config.PERTURBATIONS

    index = SubgoalIndex.load()
    bench = get_benchmark()
    runner = Runner(checkpoint,
                    eval_mode=config.eval_mode(args.condition),
                    subgoals_path=config.SUBGOALS_FILE,
                    n_action_steps=config.N_ACTION_STEPS)

    folder = os.path.join(config.PERTURB_DIR, args.condition)
    os.makedirs(folder, exist_ok=True)

    summary = {}
    for perturbation in perturbations:
        # A fresh random seed per perturbation, so runs are reproducible.
        random_state = np.random.RandomState(0)
        results = []
        print("\n=== %s: %s ===" % (perturbation, DESCRIPTIONS[perturbation]))
        for task_id in tasks:
            for episode in range(args.episodes):
                row = run_episode(runner, index, bench, task_id, episode,
                                  perturbation, random_state, args.perturb_at)
                results.append(row)
                print("  task %d ep %d  %-7s steps=%3d back=%d"
                      % (task_id, episode, "ok" if row["success"] else "fail",
                         row["steps_run"], row["n_backward"]))

        path = os.path.join(folder, "%s.jsonl" % perturbation)
        with open(path, "w") as f:
            for row in results:
                f.write(json.dumps(row) + "\n")

        successes = sum(1 for r in results if r["success"])
        backwards = [r["n_backward"] for r in results]
        summary[perturbation] = (successes, len(results), float(np.mean(backwards)))
        print("  -> %d/%d succeeded, %.2f backward steps on average -> %s"
              % (successes, len(results), np.mean(backwards), path))

    print("\n%-14s %10s %16s" % ("perturbation", "success", "backward steps"))
    for perturbation in perturbations:
        successes, total, backwards = summary[perturbation]
        print("%-14s %6d/%-4d %14.2f" % (perturbation, successes, total, backwards))

    print("\nRemember: success alone is only half the story. The other half is whether")
    print("System 2 NOTICED, which is the backward-step count. Compare against A_frozen,")
    print("which follows a clock and cannot go backwards at all.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
