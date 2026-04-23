"""Skills package for RealRobotCLI."""

from skills.detect import DetectSkill
from skills.move import MoveSkill
from skills.grasp import GraspSkill
from skills.release import ReleaseSkill
from skills.home import HomeSkill
from skills.status import StatusSkill
from skills.close_gripper import CloseGripperSkill
from skills.open_gripper import OpenGripperSkill
from skills.lift import LiftSkill

SKILLS = {
    "detect": DetectSkill,
    "move": MoveSkill,
    "grasp": GraspSkill,
    "release": ReleaseSkill,
    "home": HomeSkill,
    "status": StatusSkill,
    "close_gripper": CloseGripperSkill,
    "open_gripper": OpenGripperSkill,
    "lift": LiftSkill,
}
