"""G1-specific configurations for multi-motion distillation (blind)."""

from isaaclab.utils import configclass

from whole_body_tracking.robots.g1 import G1_ACTION_SCALE, G1_ANCHOR_BODY_NAME, G1_BODY_NAMES, G1_CYLINDER_CFG
from whole_body_tracking.tasks.tracking.config.g1.agents.rsl_rl_ppo_cfg import LOW_FREQ_SCALE
from whole_body_tracking.tasks.tracking.tracking_env_cfg import DistillationEnvCfg


@configclass
class G1DistillationEnvCfg(DistillationEnvCfg):
    """Configuration for G1 multi-motion distillation (blind, teacher-student)."""

    def __post_init__(self):
        super().__post_init__()

        # Set G1 robot and action scale
        self.scene.robot = G1_CYLINDER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.actions.joint_pos.scale = G1_ACTION_SCALE

        # Set body tracking for terminations
        self.commands.motion.anchor_body_name = G1_ANCHOR_BODY_NAME
        self.commands.motion.body_names = G1_BODY_NAMES


@configclass
class G1DistillationWoStateEstimationEnvCfg(G1DistillationEnvCfg):
    """G1 distillation without state estimation (no base velocity)."""

    def __post_init__(self):
        super().__post_init__()
        # Remove state estimation from teacher observations
        self.observations.policy.base_lin_vel = None


@configclass
class G1DistillationLowFreqEnvCfg(G1DistillationEnvCfg):
    """G1 distillation with low-frequency control (50% decimation)."""

    def __post_init__(self):
        super().__post_init__()
        self.decimation = round(self.decimation / LOW_FREQ_SCALE)
        self.rewards.action_rate_l2.weight *= LOW_FREQ_SCALE
