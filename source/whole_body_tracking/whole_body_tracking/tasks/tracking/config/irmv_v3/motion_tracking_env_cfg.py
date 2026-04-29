"""IRMV V3-specific configurations for single motion whole-body tracking."""

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

import whole_body_tracking.tasks.tracking.mdp as mdp
from whole_body_tracking.robots.irmv_v3 import (
    IRMV_V3_ACTION_SCALE,
    IRMV_V3_ANCHOR_BODY_NAME,
    IRMV_V3_BODY_NAMES,
    IRMV_V3_CFG,
)
from whole_body_tracking.tasks.tracking.tracking_env_cfg import MotionTrackingEnvCfg


@configclass
class IRMVMotionTrackingEnvCfg(MotionTrackingEnvCfg):
    """Configuration for IRMV V3 single motion whole-body tracking (blind, actor-critic)."""

    def __post_init__(self):
        super().__post_init__()

        # Set IRMV V3 robot and action scale
        self.scene.robot = IRMV_V3_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.actions.joint_pos.scale = IRMV_V3_ACTION_SCALE

        # Set motion tracking body names for IRMV V3
        self.commands.motion.anchor_body_name = IRMV_V3_ANCHOR_BODY_NAME
        self.commands.motion.body_names = IRMV_V3_BODY_NAMES

        # Override undesired_contacts reward: use elbow instead of wrist for arm end-effectors
        self.rewards.undesired_contacts = RewTerm(
            func=mdp.undesired_contacts,
            weight=-0.1,
            params={
                "sensor_cfg": SceneEntityCfg(
                    "contact_forces",
                    body_names=[
                        r"^(?!left_ankle_roll_link$)(?!right_ankle_roll_link$)(?!left_elbow_link$)(?!right_elbow_link$).+$"
                    ],
                ),
                "threshold": 1.0,
            },
        )

        # Override ee_body_pos termination: use elbow instead of wrist for arm end-effectors
        self.terminations.ee_body_pos = DoneTerm(
            func=mdp.bad_motion_body_pos_z_only,
            params={
                "command_name": "motion",
                "threshold": 0.25,
                "body_names": [
                    "left_ankle_roll_link",
                    "right_ankle_roll_link",
                    "left_elbow_link",
                    "right_elbow_link",
                ],
            },
        )
