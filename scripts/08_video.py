"""Step 8: record rollout videos with System 2's choice drawn on every frame.

    python scripts/08_video.py --condition A_learned
    python scripts/08_video.py --condition A_oracle --tasks 0,3,9 --episodes 2
    python scripts/08_video.py --condition A_learned --pairs 3:4,9:2,2:5

This is also the ONLY way to run the A_oracle condition, because the oracle
needs the true progress out of the simulator and the standard evaluator never
gives the policy access to that.

WHY NOT JUST USE lerobot-eval's VIDEOS
-------------------------------------
The normal evaluator records the cameras and nothing else. It never sees which
sub-goal was chosen, so it cannot draw it. These videos show two readouts on
every frame:

    what System 2 chose
    what was actually true (from the oracle)

Having both is what makes the bottleneck question answerable by eye. If the two
agree and the robot still fails, that is System 1 failing to execute. If they
disagree before the failure, that is System 2 reasoning badly.

WHY EACH READOUT SHOWS A PHASE AND A SKILL
------------------------------------------
They are different integers and the difference decides the diagnosis.

The PHASE is where you are inside this task: 0, 1, 2, 3. It is what System 2
classifies, and it is ordered, so forward and backward transitions are countable.

The SKILL is what the sub-goal MEANS once task identity and ordinals are merged
away. It is the row of the embedding table, so it is the only one of the two that
System 1 ever sees.

On task 8 ("put both moka pots on the stove") the phases map to skills
[9, 10, 9, 10], because the first and the second moka pot are the same skill. So
System 2 can name the wrong PHASE and still hand System 1 the right SKILL. Score
phases alone and that step counts as a reasoning error when the interface was in
fact correct. Both figures are recorded, and the overlay distinguishes the case
with its own colour.

Each video is saved with a .json file next to it, holding the phase, the skill,
and both agreement flags at every step, so the numbers can be recomputed without
re-running anything.
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config                                                    # noqa: E402
from common import (SubgoalIndex, SubgoalOracle, check_predicates,    # noqa: E402
                    get_benchmark, goal_predicates, make_env, reset_to_start)
from rollout import Runner, count_transitions                    # noqa: E402

MAX_STEPS = 520

# The progress strip plus four full-width readout lines. Each readout takes two
# lines, the identity and then the name on a line of its own, because the longest
# sub-goal ("place the black bowl in the bottom drawer of the cabinet") is 56
# characters and fits the full 512 px footer but nothing narrower. Squeezing the
# name onto the identity line truncates exactly the long names, which are the
# ones a reader cannot reconstruct from context.
FOOTER_HEIGHT = 110

# RGB, converted once at write time.
WHITE = (255, 255, 255)
GREY = (170, 170, 170)
GREEN = (80, 255, 80)       # System 2 agrees with the truth
AMBER = (255, 180, 80)      # wrong phase label, right skill: the interface was fine
RED = (255, 90, 90)         # wrong skill: System 2 handed System 1 the wrong sub-goal
CYAN = (120, 220, 255)      # System 2's own readout, and its cell in the strip


def fit(text, width_px):
    """Truncate text to what actually fits, measured rather than estimated.

    Estimating a fixed pixels-per-character errs in both directions: guess high
    and the instruction spills across the panel seam into the wrist view, guess
    low and a line of narrow glyphs is cut with 60 px of empty panel to its
    right. cv2 will measure the string, so ask it.
    """
    import cv2
    if cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)[0][0] <= width_px - 16:
        return text
    low, high = 1, len(text)
    while low < high:                       # longest prefix that still fits
        middle = (low + high + 1) // 2
        if cv2.getTextSize(text[:middle], cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)[0][0] <= width_px - 16:
            low = middle
        else:
            high = middle - 1
    return text[:low]


def draw_lines(image, lines, y0=18):
    """Write (text, colour) pairs onto an image, dark outline first so it reads
    on any background."""
    import cv2
    out = np.ascontiguousarray(image)
    y = y0
    for text, colour in lines:
        cv2.putText(out, text, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(out, text, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, colour, 1, cv2.LINE_AA)
        y += 17
    return out


def make_progress_bar(width, n_phases, chosen_phase, true_phase, phase_skills):
    """A strip of boxes, one per phase, showing both answers.

    filled cyan   what System 2 chose
    green outline what is actually true

    Each box is labelled p<phase>:s<skill>. Without the skill id the strip cannot
    show what the merged vocabulary is doing: a box is not "phase 2 of task 8", it
    is a named skill that other phases and other tasks also use, and only the id
    says so. It also lets a reader check the footer against the strip without
    opening subgoals.json.

    A box jumping backwards is the error recovery signal, and it is far easier to
    see as motion than as a digit changing.
    """
    import cv2
    bar = np.zeros((FOOTER_HEIGHT, width, 3), dtype=np.uint8)
    if n_phases <= 0:
        return bar

    padding, top, height = 8, 8, 18
    box_width = (width - 2 * padding) // n_phases
    for i in range(n_phases):
        left = padding + i * box_width
        right = left + box_width - 3

        chosen = chosen_phase is not None and i == chosen_phase
        cv2.rectangle(bar, (left, top), (right, top + height),
                      CYAN if chosen else (60, 60, 60), -1)
        # The truth is an outline so it can sit on top of the filled cell and
        # still be distinguishable: agreement looks like an outlined fill.
        if true_phase is not None and i == true_phase:
            cv2.rectangle(bar, (left, top), (right, top + height), GREEN, 2)

        label = "p%d:s%d" % (i, phase_skills[i]) if phase_skills else "p%d" % i
        ink = (0, 0, 0) if chosen else (210, 210, 210)   # black reads on the bright fill
        cv2.putText(bar, label, (left + 5, top + height - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, ink, 1, cv2.LINE_AA)
    return bar


def record_episode(runner, index, bench, condition, task_id, episode, folder, size, fps, label):
    """Run one episode and write an MP4 plus a JSON trace."""
    import cv2

    env = make_env(bench, task_id, height=size, width=size)
    try:
        obs = reset_to_start(env, bench, task_id, episode)
        runner.reset()

        instruction = bench.get_task(task_id).language
        predicates = goal_predicates(env)
        n_phases = index.n_phases(task_id)
        oracle = SubgoalOracle(task_id, n_phases)
        uses_oracle = runner.config.eval_mode == "oracle"

        # This task's phase -> skill map, for the strip's box labels. Task 8 gives
        # [9, 10, 9, 10], which is the merge made visible.
        phase_skills = [index.skill(task_id, phase) for phase in range(n_phases)]

        # Some predicates are true before the policy does anything: task 8's stove
        # starts on. Recorded so "goal 1/3" on a frame is not misread as progress,
        # and so a progress metric can subtract the baseline instead of assuming zero.
        predicates_at_start = check_predicates(env, predicates)

        path = os.path.join(folder, "task%d_ep%d.mp4" % (task_id, episode))
        writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps,
                                 (size * 2, size + FOOTER_HEIGHT))
        if not writer.isOpened():
            raise SystemExit("could not open a video writer at %s" % path)

        trace = []
        chosen_phases = []
        success = False
        steps_run = MAX_STEPS

        for step in range(MAX_STEPS):
            truth = check_predicates(env, predicates)
            n_satisfied = sum(truth)

            # Ground truth for THIS observation, worked out before the policy acts
            # so the oracle condition conditions on the current state and not the
            # previous one.
            true_phase = oracle(truth, obs["robot0_gripper_qpos"])
            true_skill = index.skill(task_id, true_phase)
            if uses_oracle:
                runner.set_oracle_phase(true_phase)

            action = runner.act(obs, instruction)

            choice = runner.last_choice()
            chosen_phase = choice["phase"] if choice else None
            chosen_skill = choice["skill"] if choice else None
            if chosen_phase is not None:
                chosen_phases.append(chosen_phase)

            # TWO agreements, because under a merged vocabulary they differ. The
            # phase one answers "did System 2 label the state correctly". The skill
            # one answers "did System 1 receive the right conditioning", which is
            # the question that decides whether a failure is reasoning or execution.
            agree_phase = chosen_phase == true_phase if chosen_phase is not None else None
            agree_skill = chosen_skill == true_skill if chosen_skill is not None else None

            # LIBERO renders upside down compared with how we want to watch it.
            scene = obs["agentview_image"][::-1, ::-1].copy()
            wrist = obs["robot0_eye_in_hand_image"][::-1, ::-1].copy()

            scene = draw_lines(scene, [
                (fit(instruction, size), WHITE),
                # The task id identifies the EPISODE, not the sub-goal, so it
                # belongs up here. A folder of clips is unreadable without it.
                ("task %d  ep %d  step %d" % (task_id, episode, step), GREY),
                (fit("%s  %d skills  goal %d/%d"
                     % (label, index.n_skills, n_satisfied, len(predicates)), size),
                 GREEN if n_satisfied else AMBER),
            ])
            wrist = draw_lines(wrist, [("wrist", WHITE)] + [
                (fit("%s %s" % ("x" if ok else ".", predicate), size),
                 GREEN if ok else GREY)
                for predicate, ok in zip(predicates, truth)
            ])

            if chosen_phase is None:
                s2_head, s2_name = "S2      (none: unconditioned policy)", ""
            else:
                s2_head = "S2      phase %d   skill %d" % (chosen_phase, chosen_skill)
                s2_name = "    %s" % index.name(task_id, chosen_phase)
            true_head = "oracle  phase %d   skill %d" % (true_phase, true_skill)
            true_name = "    %s" % index.name(task_id, true_phase)

            # Three states, not two. Collapsing the middle one into "wrong" paints
            # a correctly conditioned step red, which inverts the diagnosis on any
            # task whose phases share a skill.
            if agree_phase:
                colour = GREEN                      # both match
            elif agree_skill:
                colour = AMBER                      # wrong label, right conditioning
            elif agree_phase is False:
                colour = RED                        # wrong sub-goal delivered
            else:
                colour = GREY                       # nothing to compare

            footer = make_progress_bar(size * 2, n_phases, chosen_phase, true_phase,
                                       phase_skills)
            footer = draw_lines(footer, [
                (fit(s2_head, size * 2), CYAN),
                (fit(s2_name, size * 2), CYAN),
                (fit(true_head, size * 2), colour),
                (fit(true_name, size * 2), colour),
            ], y0=44)

            frame = np.concatenate([np.concatenate([scene, wrist], axis=1), footer], axis=0)
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

            trace.append({"step": step,
                          "chosen_phase": chosen_phase, "chosen_skill": chosen_skill,
                          "true_phase": true_phase, "true_skill": true_skill,
                          "agree_phase": agree_phase, "agree_skill": agree_skill,
                          "n_predicates_satisfied": n_satisfied})

            obs, _, done, _ = env.step(action.tolist())

            if all(check_predicates(env, predicates)):
                success = True
                steps_run = step + 1
                break
            if done:
                steps_run = step + 1
                break

        writer.release()

        forwards, backwards = count_transitions(chosen_phases)

        def rate(key):
            """Fraction of the steps where that flag was true, ignoring the steps
            where it could not be worked out."""
            flags = [t[key] for t in trace if t[key] is not None]
            return sum(flags) / len(flags) if flags else None

        record = {
            "condition": condition,
            # Where the sub-goal came from. Recorded separately from the condition
            # name because several conditions share one checkpoint and differ only
            # here, so the name alone does not say what was measured.
            "eval_mode": runner.config.eval_mode,
            "label": label,                  # the name drawn on the frame
            "task_id": task_id,
            "episode": episode,
            "instruction": instruction,
            "success": success,
            "steps_run": steps_run,
            "predicates": [str(p) for p in predicates],
            "predicates_true_at_start": predicates_at_start,
            "n_skills": index.n_skills,
            "phase_skills": phase_skills,
            "n_forward": forwards,
            "n_backward": backwards,
            # The number that separates a reasoning failure from an execution one,
            # at both levels. The skill figure is always the higher of the two, and
            # the gap is how much of an apparent reasoning error the merged
            # vocabulary absorbs.
            "agreement_with_truth": rate("agree_phase"),
            "skill_agreement_with_truth": rate("agree_skill"),
            "phase_at_end": chosen_phases[-1] if chosen_phases else None,
            "trace": trace,
        }
        with open(path.replace(".mp4", ".json"), "w") as f:
            json.dump(record, f)

        def percent(value):
            return "%.0f%%" % (100 * value) if value is not None else "-"

        print("  task %d ep %d  %-4s steps %3d  forward %d  back %d  "
              "agreement phase %s skill %s"
              % (task_id, episode, "ok" if success else "fail", steps_run,
                 forwards, backwards,
                 percent(record["agreement_with_truth"]),
                 percent(record["skill_agreement_with_truth"])))
        return record
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", default="A_learned", choices=sorted(config.CONDITIONS))
    parser.add_argument("--tasks", default=None, help="comma-separated task ids")
    parser.add_argument("--episodes", type=int, default=3, help="episodes per task")
    # --episodes always counts from 0, which is right for a campaign and wrong for
    # cherry-picking: re-rendering episode 8 of one task would mean running nine
    # episodes and throwing eight away. Episode indices are seeds, so naming them
    # individually reproduces exactly the clip a campaign already measured.
    parser.add_argument("--pairs", default=None, metavar="T:E,T:E",
                        help="record exactly these task:episode pairs, e.g. 3:4,9:2. "
                             "Overrides --tasks and --episodes.")
    parser.add_argument("--label", default=None,
                        help="name to draw on the frame instead of the condition name. "
                             "Overlay only: every recorded field keeps the real name.")
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--fps", type=int, default=20)
    args = parser.parse_args()

    checkpoint = config.checkpoint_path(args.condition)
    if not os.path.isdir(checkpoint):
        print("no checkpoint at %s" % checkpoint)
        return 1

    # The (task, episode) work list.
    if args.pairs:
        work = [tuple(int(x) for x in pair.split(":")) for pair in args.pairs.split(",")]
    else:
        tasks = ([int(t) for t in args.tasks.split(",")] if args.tasks
                 else list(range(config.N_TASKS)))
        work = [(task_id, episode) for task_id in tasks for episode in range(args.episodes)]

    index = SubgoalIndex.load()
    bench = get_benchmark()
    runner = Runner(checkpoint,
                    eval_mode=config.eval_mode(args.condition),
                    subgoals_path=config.SUBGOALS_FILE,
                    n_action_steps=config.N_ACTION_STEPS)

    folder = os.path.join(config.VIDEO_DIR, args.condition)
    os.makedirs(folder, exist_ok=True)

    print("condition %s, sub-goal source '%s'"
          % (args.condition, config.eval_mode(args.condition)))
    print("writing to %s\n" % folder)

    records = []
    for task_id, episode in work:
        records.append(record_episode(runner, index, bench, args.condition, task_id,
                                      episode, folder, args.size, args.fps,
                                      args.label or args.condition))

    successes = sum(1 for r in records if r["success"])
    phase_rates = [r["agreement_with_truth"] for r in records
                   if r["agreement_with_truth"] is not None]
    skill_rates = [r["skill_agreement_with_truth"] for r in records
                   if r["skill_agreement_with_truth"] is not None]

    print("\n%d/%d succeeded" % (successes, len(records)))
    if phase_rates:
        print("System 2 matched the truth on %.0f%% of steps by phase, %.0f%% by skill"
              % (100 * sum(phase_rates) / len(phase_rates),
                 100 * sum(skill_rates) / len(skill_rates)))
        print("\nHow to read that: high agreement together with failure means System 1")
        print("could not execute. Disagreement before a failure means System 2 reasoned")
        print("badly. Watch the gap between the two figures too: where the skill number")
        print("is much higher, System 2 mislabelled the phase but still handed System 1")
        print("the right sub-goal, so that is an execution failure and not a reasoning one.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
