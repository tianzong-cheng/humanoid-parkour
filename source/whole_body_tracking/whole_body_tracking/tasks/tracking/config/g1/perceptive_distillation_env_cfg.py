"""G1-specific configurations for multi-motion distillation with perception."""

from isaaclab.utils import configclass

from whole_body_tracking.robots.g1 import G1_ACTION_SCALE, G1_CYLINDER_CFG
from whole_body_tracking.tasks.tracking.config.g1.agents.rsl_rl_ppo_cfg import LOW_FREQ_SCALE
from whole_body_tracking.tasks.tracking.tracking_env_cfg import PerceptiveDistillationEnvCfg


@configclass
class G1PerceptiveDistillationEnvCfg(PerceptiveDistillationEnvCfg):
    """Configuration for G1 multi-motion distillation with perception (teacher-student + depth camera)."""

    def __post_init__(self):
        super().__post_init__()

        # Set G1 robot and action scale
        self.scene.robot = G1_CYLINDER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.actions.joint_pos.scale = G1_ACTION_SCALE


@configclass
class G1PerceptiveDistillationWoStateEstimationEnvCfg(G1PerceptiveDistillationEnvCfg):
    """G1 perceptive distillation without state estimation (no base velocity)."""

    def __post_init__(self):
        super().__post_init__()
        # Remove state estimation from teacher observations
        self.observations.policy.base_lin_vel = None


@configclass
class G1PerceptiveDistillationLowFreqEnvCfg(G1PerceptiveDistillationEnvCfg):
    """G1 perceptive distillation with low-frequency control (50% decimation)."""

    def __post_init__(self):
        super().__post_init__()
        self.decimation = round(self.decimation / LOW_FREQ_SCALE)
        self.rewards.action_rate_l2.weight *= LOW_FREQ_SCALE
