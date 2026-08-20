"""Input and output processing for the policy.

The function name below is found by LeRobot from the policy type name, so
renaming it breaks checkpoint loading with a confusing error.

The pipelines are exactly ACT's. The sub-goal is a whole number used to look up
a row in a table, not a sensor reading, so there is nothing to normalise about
it and it needs no entry in the dataset statistics. That is a quiet advantage of
the discrete interface I chose over a continuous one: the ablation conditions can
be swapped without touching any of this.
"""

from lerobot.processor import make_default_pre_post_processors


def make_subgoal_act_pre_post_processors(config, dataset_stats=None):
    return make_default_pre_post_processors(config, dataset_stats, normalizer_device=config.device)
