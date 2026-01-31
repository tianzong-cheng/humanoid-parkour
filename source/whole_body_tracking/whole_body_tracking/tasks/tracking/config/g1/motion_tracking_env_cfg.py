"""G1-specific configurations for single motion whole-body tracking."""

from isaaclab.utils import configclass

from whole_body_tracking.robots.g1 import G1_ACTION_SCALE, G1_ANCHOR_BODY_NAME, G1_BODY_NAMES, G1_CYLINDER_CFG
from whole_body_tracking.tasks.tracking.config.g1.agents.rsl_rl_ppo_cfg import LOW_FREQ_SCALE
from whole_body_tracking.tasks.tracking.tracking_env_cfg import MotionTrackingEnvCfg


@configclass
class G1MotionTrackingEnvCfg(MotionTrackingEnvCfg):
    """Configuration for G1 single motion whole-body tracking (blind, actor-critic)."""

    def __post_init__(self):
        super().__post_init__()

        # Set G1 robot and action scale
        self.scene.robot = G1_CYLINDER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.actions.joint_pos.scale = G1_ACTION_SCALE

        # Set motion tracking body names for G1
        self.commands.motion.anchor_body_name = G1_ANCHOR_BODY_NAME
        self.commands.motion.body_names = G1_BODY_NAMES


@configclass
class G1MotionTrackingWoStateEstimationEnvCfg(G1MotionTrackingEnvCfg):
    """G1 motion tracking without state estimation (no anchor position, no base velocity)."""

    def __post_init__(self):
        super().__post_init__()
        self.observations.policy.motion_anchor_pos_b = None
        self.observations.policy.base_lin_vel = None


@configclass
class G1MotionTrackingLowFreqEnvCfg(G1MotionTrackingEnvCfg):
    """G1 motion tracking with low-frequency control (50% decimation)."""

    def __post_init__(self):
        super().__post_init__()
        self.decimation = round(self.decimation / LOW_FREQ_SCALE)
        self.rewards.action_rate_l2.weight *= LOW_FREQ_SCALE
