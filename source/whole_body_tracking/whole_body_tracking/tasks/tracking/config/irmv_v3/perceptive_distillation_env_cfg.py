"""IRMV V3-specific configurations for perceptive distillation."""

import math

from isaaclab.sensors import RayCasterCameraCfg
from isaaclab.utils import configclass

from whole_body_tracking.robots.irmv_v3 import (
    IRMV_V3_ACTION_SCALE,
    IRMV_V3_ANCHOR_BODY_NAME,
    IRMV_V3_BODY_NAMES,
    IRMV_V3_CFG,
)
from whole_body_tracking.tasks.tracking.config.irmv_v3.agents.rsl_rl_ppo_cfg import LOW_FREQ_SCALE
from whole_body_tracking.tasks.tracking.tracking_env_cfg import PerceptiveDistillationEnvCfg


@configclass
class IRMV_V3_PerceptiveDistillationEnvCfg(PerceptiveDistillationEnvCfg):
    """Configuration for IRMV V3 perceptive distillation (teacher-student + depth camera)."""

    def __post_init__(self):
        super().__post_init__()

        # Set IRMV V3 robot and action scale
        self.scene.robot = IRMV_V3_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.actions.joint_pos.scale = IRMV_V3_ACTION_SCALE

        # Set motion tracking body names for IRMV V3
        self.commands.motion.anchor_body_name = IRMV_V3_ANCHOR_BODY_NAME
        self.commands.motion.body_names = IRMV_V3_BODY_NAMES

        # Override camera offset for IRMV V3 torso geometry
        # TODO: Adjust these values based on actual IRMV V3 camera mounting position
        if hasattr(self.scene, "tiled_camera") and self.scene.tiled_camera is not None:
            self.scene.tiled_camera.prim_path = "{ENV_REGEX_NS}/Robot/torso_link"
            # Placeholder camera offset - adjust based on actual robot
            self.scene.tiled_camera.offset = RayCasterCameraCfg.OffsetCfg(
                pos=(0.05, 0.0, 0.45),
                rot=(math.cos(math.radians(48) / 2), 0.0, math.sin(math.radians(48) / 2), 0.0),
                convention="world",
            )


@configclass
class IRMV_V3_PerceptiveDistillationWoStateEstimationEnvCfg(IRMV_V3_PerceptiveDistillationEnvCfg):
    """IRMV V3 perceptive distillation without state estimation."""

    def __post_init__(self):
        super().__post_init__()
        self.observations.teacher.motion_anchor_pos_b = None
        self.observations.teacher.base_lin_vel = None
        self.observations.student.base_lin_vel = None


@configclass
class IRMV_V3_PerceptiveDistillationLowFreqEnvCfg(IRMV_V3_PerceptiveDistillationEnvCfg):
    """IRMV V3 perceptive distillation with low-frequency control."""

    def __post_init__(self):
        super().__post_init__()
        self.decimation = round(self.decimation / LOW_FREQ_SCALE)
