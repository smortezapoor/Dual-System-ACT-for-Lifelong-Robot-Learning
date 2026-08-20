"""Step 0: check the installation works, before spending hours on training.

    export MUJOCO_GL=egl
    python scripts/00_check_setup.py

Five checks, cheapest first, each explaining what to do if it fails. Every one
of these caught a real problem at some point, and every one of them fails in a
way that is confusing if you meet it later instead of here.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def check_rendering():
    """Can MuJoCo draw a picture without a screen?

    Everything else depends on this. It needs MUJOCO_GL=egl set in the SHELL,
    before Python starts, because the setting is read when mujoco is imported.
    Setting it inside Python after the import does nothing at all, and the
    failure shows up much later as blank images.
    """
    setting = os.environ.get("MUJOCO_GL")
    if setting != "egl":
        print("  MUJOCO_GL is %r, it should be 'egl'." % setting)
        print("  Run this in your shell, then try again:  export MUJOCO_GL=egl")
        return False

    import mujoco
    import numpy as np

    model_xml = """
    <mujoco>
      <worldbody>
        <light pos="0 0 3" dir="0 0 -1"/>
        <geom type="plane" size="5 5 .1" rgba=".8 .8 .8 1"/>
        <body pos="0 0 1">
          <joint type="free"/>
          <geom type="sphere" size=".3" rgba=".9 .3 .2 1"/>
        </body>
      </worldbody>
    </mujoco>
    """
    model = mujoco.MjModel.from_xml_string(model_xml)
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=256, width=256)
    mujoco.mj_forward(model, data)
    renderer.update_scene(data)
    image = renderer.render()

    if image.shape != (256, 256, 3):
        print("  wrong image shape: %s" % (image.shape,))
        return False
    if int(np.asarray(image).std()) == 0:
        print("  the image is blank, so rendering is not really working")
        return False
    print("  rendered a 256x256 image")
    return True


def check_lerobot():
    """Is LeRobot installed, and does it know about our policy?

    LeRobot only looks for out-of-tree policies inside its own commands, so a
    plain script has to ask it to look. If our policy is missing here, the usual
    cause is the package name being written with hyphens instead of underscores,
    which makes the import fail silently.
    """
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.utils.import_utils import register_third_party_plugins

    register_third_party_plugins()
    known = PreTrainedConfig.get_known_choices()

    if "subgoal_act" not in known:
        print("  'subgoal_act' is not registered. Known policies: %s" % sorted(known))
        print("  Install the plugin:  pip install -e ./policy")
        print("  If it is installed, check the name in policy/pyproject.toml uses")
        print("  UNDERSCORES: lerobot_policy_subgoal_act, not hyphens.")
        return False

    print("  LeRobot knows about 'subgoal_act' (%d policies available)" % len(known))
    return True


def check_subgoals():
    """Does subgoals.json make sense?"""
    from common import SubgoalIndex

    index = SubgoalIndex.load()
    if len(index.tasks) != 10:
        print("  expected 10 tasks, found %d" % len(index.tasks))
        return False

    used = set()
    for task_id in index.tasks:
        for phase in range(index.n_phases(task_id)):
            used.add(index.skill(task_id, phase))

    # Skill ids must run 0, 1, 2, ... with no gaps. A gap means an embedding row
    # that is never used, and a table sized wrongly.
    if used != set(range(index.n_skills)):
        print("  skill ids have gaps: expected 0..%d" % (index.n_skills - 1))
        return False

    print("  10 tasks, %d skills, ids are complete" % index.n_skills)
    return True


def check_both_index_copies_agree():
    """The plugin has its own copy of the index. They must not drift apart.

    The plugin needs to work when installed on its own, without this repo, and
    the scripts need to work without the plugin installed. So neither can import
    the other, and there are two copies. If they ever disagree, conditioning
    would quietly point at the wrong embedding rows and nothing would raise.
    """
    from common import SubgoalIndex as OurIndex
    from lerobot_policy_subgoal_act.subgoal_index import SubgoalIndex as PluginIndex
    from config import SUBGOALS_FILE

    ours = OurIndex.load()
    theirs = PluginIndex.load(SUBGOALS_FILE)

    if ours.n_skills != theirs.n_skills or ours.null_skill != theirs.null_skill:
        print("  the two copies disagree on how many skills there are")
        return False

    for task_id in ours.tasks:
        if ours.n_phases(task_id) != theirs.n_phases(task_id):
            print("  the two copies disagree on task %d" % task_id)
            return False
        for phase in range(ours.n_phases(task_id)):
            if ours.skill(task_id, phase) != theirs.skill(task_id, phase):
                print("  the two copies disagree on task %d phase %d" % (task_id, phase))
                return False

    print("  both copies of the index agree on all 10 tasks")
    return True


def check_simulator():
    """Can we actually build a LIBERO task and step it?"""
    from common import get_benchmark, make_env, reset_to_start, goal_predicates, check_predicates

    bench = get_benchmark()
    env = make_env(bench, 3)
    try:
        obs = reset_to_start(env, bench, 3, 0)
        predicates = goal_predicates(env)
        truth = check_predicates(env, predicates)
        obs, _, _, _ = env.step([0.0] * 6 + [-1.0])

        print("  task 3 runs: image %s, %d goal conditions, %d true at the start"
              % (obs["agentview_image"].shape, len(predicates), sum(truth)))
        return True
    finally:
        env.close()


CHECKS = [
    ("rendering without a screen", check_rendering),
    ("subgoals.json", check_subgoals),
    ("LeRobot and the policy plugin", check_lerobot),
    ("the two index copies agree", check_both_index_copies_agree),
    ("the LIBERO simulator", check_simulator),
]


def main():
    failed = []
    for name, check in CHECKS:
        print("\n[%s]" % name)
        try:
            if not check():
                failed.append(name)
        except Exception as error:
            print("  FAILED: %s: %s" % (type(error).__name__, error))
            failed.append(name)

    print("\n" + "-" * 60)
    if failed:
        print("FAILED: %s" % ", ".join(failed))
        return 1
    print("everything works. Next:")
    print("  python scripts/01_make_subgoals.py --check")
    print("  python scripts/02_label_dataset.py")
    print("  python scripts/03_train.py A_learned --smoke")
    return 0


if __name__ == "__main__":
    sys.exit(main())
