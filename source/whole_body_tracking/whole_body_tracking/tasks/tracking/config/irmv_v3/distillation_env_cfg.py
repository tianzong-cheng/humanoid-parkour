"""IRMV V3-specific configurations for multi-motion distillation."""

from isaaclab.utils import configclass

from whole_body_tracking.robots.irmv_v3 import (
    IRMV_V3_ACTION_SCALE,
    IRMV_V3_ANCHOR_BODY_NAME,
    IRMV_V3_BODY_NAMES,
    IRMV_V3_CFG,
)
from whole_body_tracking.tasks.tracking.config.irmv_v3.agents.rsl_rl_ppo_cfg import LOW_FREQ_SCALE
from whole_body_tracking.tasks.tracking.tracking_env_cfg import DistillationEnvCfg


@configclass
class IRMV_V3_DistillationEnvCfg(DistillationEnvCfg):
    """Configuration for IRMV V3 multi-motion distillation (blind, teacher-student)."""

    def __post_init__(self):
        super().__post_init__()

        # Set IRMV V3 robot and action scale
        self.scene.robot = IRMV_V3_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.actions.joint_pos.scale = IRMV_V3_ACTION_SCALE

        # Set motion tracking body names for IRMV V3
        self.commands.motion.anchor_body_name = IRMV_V3_ANCHOR_BODY_NAME
        self.commands.motion.body_names = IRMV_V3_BODY_NAMES


@configclass
class IRMV_V3_DistillationWoStateEstimationEnvCfg(IRMV_V3_DistillationEnvCfg):
    """IRMV V3 distillation without state estimation."""

    def __post_init__(self):
        super().__post_init__()
        self.observations.teacher.motion_anchor_pos_b = None
        self.observations.teacher.base_lin_vel = None
        self.observations.student.base_lin_vel = None


@configclass
class IRMV_V3_DistillationLowFreqEnvCfg(IRMV_V3_DistillationEnvCfg):
    """IRMV V3 distillation with low-frequency control (50% decimation)."""

    def __post_init__(self):
        super().__post_init__()
        self.decimation = round(self.decimation / LOW_FREQ_SCALE)
