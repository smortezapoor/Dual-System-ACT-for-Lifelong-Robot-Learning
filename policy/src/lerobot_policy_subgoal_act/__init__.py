"""Sub-goal conditioned ACT, as a LeRobot plugin.

Importing the configuration module here is what actually registers the policy
type with LeRobot. Without this line the package installs fine and
"--policy.type=subgoal_act" stays unknown.
"""

from .configuration_subgoal_act import SubgoalACTConfig
from .subgoal_index import SubgoalIndex

__all__ = ["SubgoalACTConfig", "SubgoalIndex"]
